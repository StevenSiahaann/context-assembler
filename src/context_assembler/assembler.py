"""Context Assembler pipeline orchestrator."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from context_assembler.allocation.water_fill import water_fill
from context_assembler.domain.block import (
    Alternatives,
    ContextBlock,
    HeadTail,
    MiddleOut,
    Priority,
    Summarize,
    TokenizedBlock,
)
from context_assembler.domain.budget import ModelProfile, SlotPolicy
from context_assembler.domain.errors import P0OverflowError
from context_assembler.grouping.atomic_groups import validate_atomic_groups
from context_assembler.observability.report import (
    AssemblyReport,
    BlockDecision,
    DecisionReason,
    SlotUsage,
)
from context_assembler.ordering.cache_aware import order_blocks_for_caching
from context_assembler.planning.budget_planner import BudgetPlan, calculate_input_budget
from context_assembler.preprocess.dedup import (
    apply_supersession,
    deduplicate_exact,
)
from context_assembler.preprocess.pre_clip import pre_clip_block
from context_assembler.security.trust_wrap import format_untrusted_envelope
from context_assembler.strategies.alternatives import reduce_alternatives
from context_assembler.strategies.anchored_window import reduce_anchored_window
from context_assembler.strategies.drop_lowest import reduce_drop_lowest
from context_assembler.strategies.head_tail import reduce_head_tail
from context_assembler.structure.normalize import normalize_message_structure
from context_assembler.structure.validators import validate_assembled_structure
from context_assembler.tokenizers.base import Tokenizer


@dataclass(frozen=True, slots=True)
class AssembledContext:
    """Assembled output payload containing ordered blocks and execution report."""

    blocks: list[ContextBlock]
    report: AssemblyReport


class Assembler:
    """Deterministic, cache-aware context assembler for LLM prompt construction."""

    def __init__(
        self,
        model: ModelProfile,
        tokenizer: Tokenizer,
        slots: Sequence[SlotPolicy] | None = None,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._slot_policies: dict[str, SlotPolicy] = {
            s.kind: s
            for s in (
                slots
                if slots is not None
                else [
                    SlotPolicy("tool_result", min_tokens=500, max_tokens=15_000, weight=3),
                    SlotPolicy("rag", min_tokens=1_000, max_tokens=25_000, weight=4),
                    SlotPolicy("memory", min_tokens=200, max_tokens=5_000, weight=1),
                    SlotPolicy("history", min_tokens=1_000, max_tokens=30_000, weight=2),
                ]
            )
        }

    @property
    def model(self) -> ModelProfile:
        return self._model

    def _tokenize(self, block: ContextBlock) -> TokenizedBlock:
        wrapped_content = format_untrusted_envelope(block, block.content)
        target_block = (
            replace(block, content=wrapped_content) if wrapped_content != block.content else block
        )
        if target_block.fixed_tokens is not None:
            return TokenizedBlock(target_block, target_block.fixed_tokens)
        cost = self._tokenizer.count(target_block.content)
        return TokenizedBlock(target_block, cost)

    def _reduce_single(self, tb: TokenizedBlock, budget: int) -> TokenizedBlock | None:
        """Applies block-level reduction strategy to fit within budget."""
        strat = tb.block.strategy
        if isinstance(strat, (HeadTail, MiddleOut)):
            return reduce_head_tail(tb, budget, self._tokenizer)
        if isinstance(strat, Alternatives):
            return reduce_alternatives(tb, budget, self._tokenizer)
        return None

    def _route_slot(self, block: ContextBlock) -> str:
        """Routes a dynamic block to a configured slot policy."""
        if block.kind in ("tool_call", "tool_result") and "tool_result" in self._slot_policies:
            return "tool_result"
        if block.kind in self._slot_policies:
            return block.kind
        if "history" in self._slot_policies:
            return "history"
        return next(iter(self._slot_policies))

    def _slot_order(self) -> list[str]:
        preferred_order = ("tool_result", "rag", "memory", "history")
        order = [k for k in preferred_order if k in self._slot_policies]
        for k in self._slot_policies:
            if k not in order:
                order.append(k)
        return order

    def plan(
        self,
        fixed_blocks: Sequence[ContextBlock],
        demand: Mapping[str, int] | None = None,
        estimated_messages: int = 16,
    ) -> BudgetPlan:
        """Computes a two-phase slot budget plan prior to fetching RAG or history.

        Allows callers to query `plan.slot_allocations['rag']` or
        `plan.recommended_top_k('rag')` before running expensive retrieval.
        """
        tokenized = [self._tokenize(b) for b in fixed_blocks]
        tool_count = sum(1 for tb in tokenized if tb.block.kind == "tool_schema")
        msg_estimate = max(len(tokenized), estimated_messages)
        planned = calculate_input_budget(
            self._model,
            estimated_messages=msg_estimate,
            tool_count=tool_count,
        )
        input_budget = planned.input_budget
        merge_reserve = msg_estimate if self._model.requires_alternation else 0
        usable_budget = max(0, input_budget - merge_reserve)

        p0_blocks = [tb for tb in tokenized if tb.block.priority == Priority.P0 or tb.block.pinned]
        p0_total = sum(tb.tokens for tb in p0_blocks)
        if p0_total > usable_budget:
            raise P0OverflowError(required=p0_total, available=input_budget)

        p0_ids = {id(tb) for tb in p0_blocks}
        p1_blocks = [
            tb for tb in tokenized if tb.block.priority == Priority.P1 and id(tb) not in p0_ids
        ]
        p1_total = min(usable_budget - p0_total, sum(tb.tokens for tb in p1_blocks))
        remaining = max(0, usable_budget - p0_total - p1_total)

        slot_order = self._slot_order()
        effective_demand = {
            k: (
                demand[k]
                if demand is not None and k in demand
                else self._slot_policies[k].max_tokens
            )
            for k in slot_order
        }
        allocations = water_fill(
            budget=remaining,
            demand=effective_demand,
            policy=self._slot_policies,
            order=slot_order,
        )
        return BudgetPlan(
            input_budget=input_budget,
            p0_tokens=p0_total,
            p1_tokens=p1_total,
            remaining_for_slots=remaining,
            slot_allocations=allocations,
        )

    def assemble(
        self,
        raw_blocks: Sequence[ContextBlock],
        plan: BudgetPlan | None = None,
    ) -> AssembledContext:
        """Assembles prompt context within budget constraints.

        Guarantees:
        1. Resulting total tokens <= planned input budget.
        2. Non-negotiable P0 prompts are never dropped.
        3. Atomic tool call/result pairs remain paired.
        4. Messages adhere to provider structure rules.
        """
        start_time = time.perf_counter_ns()
        decisions_by_id: dict[str, BlockDecision] = {}

        # 1. Preprocessing: coarse clipping, deduplication, state supersession
        pre_clipped = [pre_clip_block(b) for b in raw_blocks]
        deduped = deduplicate_exact(pre_clipped)
        deduped_ids = {id(b) for b in deduped}
        for b in pre_clipped:
            if id(b) not in deduped_ids:
                raw_cost = self._tokenizer.count(b.content)
                decisions_by_id[b.id] = BlockDecision(
                    block_id=b.id,
                    action="dropped",
                    tokens_before=raw_cost,
                    tokens_after=0,
                    reason="duplicate",
                )

        superseded = apply_supersession(deduped)
        superseded_ids: set[str] = {
            s.id for d, s in zip(deduped, superseded, strict=True) if d.content != s.content
        }

        # 2. Invariant: Validate atomic pairing
        validate_atomic_groups(superseded)

        # 3. Tokenize candidate blocks (includes untrusted <document> wrapper tokens)
        tokenized = [self._tokenize(b) for b in superseded]

        # 4. Budget planning
        tool_count = sum(1 for b in tokenized if b.block.kind == "tool_schema")
        planned = calculate_input_budget(
            self._model,
            estimated_messages=len(tokenized),
            tool_count=tool_count,
        )
        input_budget = planned.input_budget

        # Reserve 1 separator token per message if strict alternation may merge adjacent turns
        merge_reserve = len(tokenized) if self._model.requires_alternation else 0
        usable_budget = max(0, input_budget - merge_reserve)

        # 5. Non-negotiable P0 & pinned allocation
        p0_blocks = [tb for tb in tokenized if tb.block.priority == Priority.P0 or tb.block.pinned]
        p0_total = sum(tb.tokens for tb in p0_blocks)

        if p0_total > usable_budget:
            raise P0OverflowError(required=p0_total, available=input_budget)

        for pb in p0_blocks:
            decisions_by_id[pb.block.id] = BlockDecision(
                block_id=pb.block.id,
                action="included",
                tokens_before=pb.tokens,
                tokens_after=pb.tokens,
                reason="fits",
            )

        budget_remaining = usable_budget - p0_total
        p0_id_set = {id(pb) for pb in p0_blocks}

        # 6. Active user request (P1, ungrouped)
        p1_blocks = [
            tb
            for tb in tokenized
            if tb.block.priority == Priority.P1
            and id(tb) not in p0_id_set
            and tb.block.group_id is None
        ]
        p1_id_set = {id(tb) for tb in p1_blocks}
        included_p1: list[TokenizedBlock] = []

        for tb in p1_blocks:
            if tb.tokens <= budget_remaining:
                included_p1.append(tb)
                budget_remaining -= tb.tokens
                decisions_by_id[tb.block.id] = BlockDecision(
                    block_id=tb.block.id,
                    action="included",
                    tokens_before=tb.tokens,
                    tokens_after=tb.tokens,
                    reason="fits",
                )
            else:
                trimmed = self._reduce_single(tb, budget_remaining)
                if trimmed is not None and trimmed.tokens <= budget_remaining:
                    included_p1.append(trimmed)
                    budget_remaining -= trimmed.tokens
                    decisions_by_id[tb.block.id] = BlockDecision(
                        block_id=tb.block.id,
                        action="truncated",
                        tokens_before=tb.tokens,
                        tokens_after=trimmed.tokens,
                        reason="slot_budget_exceeded",
                    )
                else:
                    action = (
                        "needs_summary" if isinstance(tb.block.strategy, Summarize) else "dropped"
                    )
                    decisions_by_id[tb.block.id] = BlockDecision(
                        block_id=tb.block.id,
                        action=action,
                        tokens_before=tb.tokens,
                        tokens_after=0,
                        reason="slot_budget_exceeded",
                    )

        # 7. Partition remaining budget across dynamic slots via water-filling
        dynamic_candidates = [
            tb for tb in tokenized if id(tb) not in p0_id_set and id(tb) not in p1_id_set
        ]
        demand_per_slot: dict[str, int] = defaultdict(int)
        blocks_by_slot: dict[str, list[TokenizedBlock]] = defaultdict(list)

        for tb in dynamic_candidates:
            slot_kind = self._route_slot(tb.block)
            demand_per_slot[slot_kind] += tb.tokens
            blocks_by_slot[slot_kind].append(tb)

        slot_order = self._slot_order()

        if plan is not None and sum(plan.slot_allocations.values()) <= budget_remaining:
            allocation = {
                k: min(plan.slot_allocations.get(k, 0), demand_per_slot.get(k, 0))
                for k in slot_order
            }
            slack = budget_remaining - sum(allocation.values())
            if slack > 0:
                extra_demand = {
                    k: max(0, demand_per_slot.get(k, 0) - allocation.get(k, 0)) for k in slot_order
                }
                extra_alloc = water_fill(
                    budget=slack,
                    demand=extra_demand,
                    policy=self._slot_policies,
                    order=slot_order,
                )
                for k in slot_order:
                    allocation[k] = min(
                        self._slot_policies[k].max_tokens,
                        allocation[k] + extra_alloc.get(k, 0),
                    )
        else:
            allocation = water_fill(
                budget=budget_remaining,
                demand=demand_per_slot,
                policy=self._slot_policies,
                order=slot_order,
            )

        slot_usage: dict[str, SlotUsage] = {}
        included_dynamic: list[TokenizedBlock] = []

        # 8. Apply per-slot reduction strategies
        for kind in slot_order:
            slot_budget = allocation.get(kind, 0)
            cands = blocks_by_slot.get(kind, [])
            spent_in_slot = 0

            if not cands:
                slot_usage[kind] = SlotUsage(allocated=slot_budget, used=0, demand=0)
                continue

            if kind == "tool_result":
                spent_in_slot = self._reduce_tool_slot(
                    cands=cands,
                    slot_budget=slot_budget,
                    included_out=included_dynamic,
                    decisions_out=decisions_by_id,
                    superseded_ids=superseded_ids,
                )
            elif kind in ("rag", "memory"):
                inc, drp = reduce_drop_lowest(cands, slot_budget)
                for tb in inc:
                    included_dynamic.append(tb)
                    spent_in_slot += tb.tokens
                    reason: DecisionReason = (
                        "superseded" if tb.block.id in superseded_ids else "fits"
                    )
                    decisions_by_id[tb.block.id] = BlockDecision(
                        block_id=tb.block.id,
                        action="included",
                        tokens_before=tb.tokens,
                        tokens_after=tb.tokens,
                        reason=reason,
                    )
                for tb in drp:
                    rem = max(0, slot_budget - spent_in_slot)
                    trimmed = self._reduce_single(tb, rem)
                    if trimmed is not None and spent_in_slot + trimmed.tokens <= slot_budget:
                        included_dynamic.append(trimmed)
                        spent_in_slot += trimmed.tokens
                        decisions_by_id[tb.block.id] = BlockDecision(
                            block_id=tb.block.id,
                            action="truncated",
                            tokens_before=tb.tokens,
                            tokens_after=trimmed.tokens,
                            reason="slot_budget_exceeded",
                        )
                    else:
                        decisions_by_id[tb.block.id] = BlockDecision(
                            block_id=tb.block.id,
                            action="dropped",
                            tokens_before=tb.tokens,
                            tokens_after=0,
                            reason="lower_score",
                        )
            else:
                low_wm = self._slot_policies[kind].low_watermark_ratio
                inc, drp = reduce_anchored_window(
                    cands,
                    slot_budget,
                    low_watermark_ratio=low_wm,
                )
                for tb in inc:
                    included_dynamic.append(tb)
                    spent_in_slot += tb.tokens
                    decisions_by_id[tb.block.id] = BlockDecision(
                        block_id=tb.block.id,
                        action="included",
                        tokens_before=tb.tokens,
                        tokens_after=tb.tokens,
                        reason="fits",
                    )
                for tb in drp:
                    decisions_by_id[tb.block.id] = BlockDecision(
                        block_id=tb.block.id,
                        action="dropped",
                        tokens_before=tb.tokens,
                        tokens_after=0,
                        reason="outside_window",
                    )

            slot_usage[kind] = SlotUsage(
                allocated=slot_budget,
                used=spent_in_slot,
                demand=demand_per_slot.get(kind, 0),
            )

        # 9. Cache-aware ordering: prefix -> history & tool pairs -> rag -> active user
        selected_tokens = [*p0_blocks, *included_p1, *included_dynamic]
        ordered, prefix_hash, prefix_tokens = order_blocks_for_caching(selected_tokens)

        # 10. Structure normalization (guarantees first chat turn is user & alternation)
        normalized, struct_dropped = normalize_message_structure(
            ordered,
            requires_alternation=self._model.requires_alternation,
            tokenizer=self._tokenizer,
        )
        for tb in struct_dropped:
            prev_dec = decisions_by_id.get(tb.block.id)
            before = prev_dec.tokens_before if prev_dec else tb.tokens
            decisions_by_id[tb.block.id] = BlockDecision(
                block_id=tb.block.id,
                action="dropped",
                tokens_before=before,
                tokens_after=0,
                reason="structural_fix",
            )

        # 11. Final verification check
        final_total = sum(tb.tokens for tb in normalized)
        validate_assembled_structure(
            normalized,
            requires_alternation=self._model.requires_alternation,
        )

        duration_ms = (time.perf_counter_ns() - start_time) / 1_000_000.0

        tool_schema_costs = {
            tb.block.id: tb.tokens for tb in normalized if tb.block.kind == "tool_schema"
        }

        ordered_decisions = tuple(
            decisions_by_id[b.id] for b in raw_blocks if b.id in decisions_by_id
        )

        report = AssemblyReport(
            input_budget=input_budget,
            total_tokens=final_total,
            utilization=round(final_total / max(1, input_budget), 4),
            per_slot=slot_usage,
            cache_prefix_tokens=prefix_tokens,
            prefix_hash=prefix_hash,
            tool_schema_tokens=tool_schema_costs,
            warnings=() if final_total <= input_budget else ("budget_overrun_detected",),
            decisions=ordered_decisions,
            duration_ms=round(duration_ms, 2),
        )

        return AssembledContext(
            blocks=[tb.block for tb in normalized],
            report=report,
        )

    def _reduce_tool_slot(
        self,
        cands: Sequence[TokenizedBlock],
        slot_budget: int,
        included_out: list[TokenizedBlock],
        decisions_out: dict[str, BlockDecision],
        superseded_ids: set[str],
    ) -> int:
        """Reduces tool slot blocks while keeping atomic tool_call/tool_result groups intact."""
        spent = 0
        processed_groups: set[str] = set()

        for tb in cands:
            gid = tb.block.group_id
            if gid is None:
                rem = max(0, slot_budget - spent)
                if tb.tokens <= rem:
                    included_out.append(tb)
                    spent += tb.tokens
                    decisions_out[tb.block.id] = BlockDecision(
                        block_id=tb.block.id,
                        action="included",
                        tokens_before=tb.tokens,
                        tokens_after=tb.tokens,
                        reason="superseded" if tb.block.id in superseded_ids else "fits",
                    )
                else:
                    trimmed = self._reduce_single(tb, rem)
                    if trimmed is not None and spent + trimmed.tokens <= slot_budget:
                        included_out.append(trimmed)
                        spent += trimmed.tokens
                        decisions_out[tb.block.id] = BlockDecision(
                            block_id=tb.block.id,
                            action="truncated",
                            tokens_before=tb.tokens,
                            tokens_after=trimmed.tokens,
                            reason="slot_budget_exceeded",
                        )
                    else:
                        decisions_out[tb.block.id] = BlockDecision(
                            block_id=tb.block.id,
                            action="dropped",
                            tokens_before=tb.tokens,
                            tokens_after=0,
                            reason="slot_budget_exceeded",
                        )
                continue

            if gid in processed_groups:
                continue
            processed_groups.add(gid)

            group_members = [m for m in cands if m.block.group_id == gid]
            calls = [m for m in group_members if m.block.kind == "tool_call"]
            results = [m for m in group_members if m.block.kind != "tool_call"]

            rem = max(0, slot_budget - spent)
            call_cost = sum(c.tokens for c in calls)
            if call_cost > rem:
                for m in group_members:
                    decisions_out[m.block.id] = BlockDecision(
                        block_id=m.block.id,
                        action="dropped",
                        tokens_before=m.tokens,
                        tokens_after=0,
                        reason="group_dropped",
                    )
                continue

            rem_for_results = rem - call_cost
            fitted_results: list[tuple[TokenizedBlock, TokenizedBlock, bool]] = []
            group_fits = True

            for r in results:
                if r.tokens <= rem_for_results:
                    fitted_results.append((r, r, False))
                    rem_for_results -= r.tokens
                else:
                    trimmed = self._reduce_single(r, rem_for_results)
                    if trimmed is not None and trimmed.tokens <= rem_for_results:
                        fitted_results.append((r, trimmed, True))
                        rem_for_results -= trimmed.tokens
                    else:
                        group_fits = False
                        break

            if not group_fits:
                for m in group_members:
                    decisions_out[m.block.id] = BlockDecision(
                        block_id=m.block.id,
                        action="dropped",
                        tokens_before=m.tokens,
                        tokens_after=0,
                        reason="group_dropped",
                    )
                continue

            for c in calls:
                included_out.append(c)
                spent += c.tokens
                decisions_out[c.block.id] = BlockDecision(
                    block_id=c.block.id,
                    action="included",
                    tokens_before=c.tokens,
                    tokens_after=c.tokens,
                    reason="fits",
                )

            for orig_r, final_r, was_truncated in fitted_results:
                included_out.append(final_r)
                spent += final_r.tokens
                decisions_out[orig_r.block.id] = BlockDecision(
                    block_id=orig_r.block.id,
                    action="truncated" if was_truncated else "included",
                    tokens_before=orig_r.tokens,
                    tokens_after=final_r.tokens,
                    reason=(
                        "slot_budget_exceeded"
                        if was_truncated
                        else ("superseded" if orig_r.block.id in superseded_ids else "fits")
                    ),
                )

        return spent
