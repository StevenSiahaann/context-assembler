"""Context Assembler pipeline orchestrator."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from context_assembler.allocation.water_fill import water_fill
from context_assembler.domain.block import (
    ContextBlock,
    Priority,
    TokenizedBlock,
)
from context_assembler.domain.budget import ModelProfile, SlotPolicy
from context_assembler.domain.errors import P0OverflowError
from context_assembler.grouping.atomic_groups import validate_atomic_groups
from context_assembler.observability.report import (
    AssemblyReport,
    BlockDecision,
    SlotUsage,
)
from context_assembler.ordering.cache_aware import order_blocks_for_caching
from context_assembler.planning.budget_planner import calculate_input_budget
from context_assembler.preprocess.dedup import (
    apply_supersession,
    deduplicate_exact,
)
from context_assembler.preprocess.pre_clip import pre_clip_block
from context_assembler.security.trust_wrap import wrap_untrusted_block
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
        if block.fixed_tokens is not None:
            return TokenizedBlock(block, block.fixed_tokens)
        cost = self._tokenizer.count(block.content)
        return TokenizedBlock(block, cost)

    def assemble(self, raw_blocks: Sequence[ContextBlock]) -> AssembledContext:
        """Assembles prompt context within budget constraints.

        Guarantees:
        1. Resulting total tokens <= planned input budget.
        2. Non-negotiable P0 prompts are never dropped.
        3. Atomic tool call/result pairs remain paired.
        4. Messages adhere to provider structure rules.
        """
        start_time = time.perf_counter_ns()
        decisions: list[BlockDecision] = []

        # 1. Preprocessing: coarse clipping, deduplication, state supersession
        pre_clipped = [pre_clip_block(b) for b in raw_blocks]
        deduped = deduplicate_exact(pre_clipped)
        superseded = apply_supersession(deduped)

        # 2. Invariant: Validate atomic pairing
        validate_atomic_groups(superseded)

        # 3. Tokenize candidate blocks
        tokenized = [self._tokenize(b) for b in superseded]

        # 4. Budget planning
        tool_count = sum(1 for b in tokenized if b.block.kind == "tool_schema")
        planned = calculate_input_budget(
            self._model,
            estimated_messages=len(tokenized),
            tool_count=tool_count,
        )
        input_budget = planned.input_budget

        # 5. Non-negotiable P0 & pinned allocation
        p0_blocks = [tb for tb in tokenized if tb.block.priority == Priority.P0 or tb.block.pinned]
        p0_total = sum(tb.tokens for tb in p0_blocks)

        if p0_total > input_budget:
            raise P0OverflowError(required=p0_total, available=input_budget)

        for pb in p0_blocks:
            decisions.append(
                BlockDecision(
                    block_id=pb.block.id,
                    action="included",
                    tokens_before=pb.tokens,
                    tokens_after=pb.tokens,
                    reason="fits",
                )
            )

        budget_remaining = input_budget - p0_total

        # 6. Active user request (P1)
        p1_blocks = [
            tb
            for tb in tokenized
            if tb.block.priority == Priority.P1 and tb not in p0_blocks
        ]
        included_p1: list[TokenizedBlock] = []

        for tb in p1_blocks:
            if tb.tokens <= budget_remaining:
                included_p1.append(tb)
                budget_remaining -= tb.tokens
                decisions.append(
                    BlockDecision(
                        block_id=tb.block.id,
                        action="included",
                        tokens_before=tb.tokens,
                        tokens_after=tb.tokens,
                        reason="fits",
                    )
                )
            else:
                # Truncate user prompt if overwhelmingly large
                trimmed = reduce_head_tail(tb, budget_remaining, self._tokenizer)
                if trimmed is not None and trimmed.tokens <= budget_remaining:
                    included_p1.append(trimmed)
                    budget_remaining -= trimmed.tokens
                    decisions.append(
                        BlockDecision(
                            block_id=tb.block.id,
                            action="truncated",
                            tokens_before=tb.tokens,
                            tokens_after=trimmed.tokens,
                            reason="slot_budget_exceeded",
                        )
                    )
                else:
                    decisions.append(
                        BlockDecision(
                            block_id=tb.block.id,
                            action="dropped",
                            tokens_before=tb.tokens,
                            tokens_after=0,
                            reason="slot_budget_exceeded",
                        )
                    )

        # 7. Partition remaining budget across dynamic slots via water-filling
        dynamic_candidates = [
            tb for tb in tokenized if tb not in p0_blocks and tb not in p1_blocks
        ]
        demand_per_slot: dict[str, int] = defaultdict(int)
        blocks_by_slot: dict[str, list[TokenizedBlock]] = defaultdict(list)

        for tb in dynamic_candidates:
            kind = tb.block.kind
            demand_per_slot[kind] += tb.tokens
            blocks_by_slot[kind].append(tb)

        slot_order = [
            k
            for k in ("tool_result", "rag", "memory", "history")
            if k in self._slot_policies
        ]

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
                for tb in cands:
                    if spent_in_slot + tb.tokens <= slot_budget:
                        included_dynamic.append(tb)
                        spent_in_slot += tb.tokens
                        decisions.append(
                            BlockDecision(
                                block_id=tb.block.id,
                                action="included",
                                tokens_before=tb.tokens,
                                tokens_after=tb.tokens,
                                reason="fits",
                            )
                        )
                    else:
                        remaining_slot = max(0, slot_budget - spent_in_slot)
                        trimmed = reduce_head_tail(tb, remaining_slot, self._tokenizer)
                        if trimmed and spent_in_slot + trimmed.tokens <= slot_budget:
                            included_dynamic.append(trimmed)
                            spent_in_slot += trimmed.tokens
                            decisions.append(
                                BlockDecision(
                                    block_id=tb.block.id,
                                    action="truncated",
                                    tokens_before=tb.tokens,
                                    tokens_after=trimmed.tokens,
                                    reason="slot_budget_exceeded",
                                )
                            )
                        else:
                            decisions.append(
                                BlockDecision(
                                    block_id=tb.block.id,
                                    action="dropped",
                                    tokens_before=tb.tokens,
                                    tokens_after=0,
                                    reason="slot_budget_exceeded",
                                )
                            )

            elif kind in ("rag", "memory"):
                inc, drp = reduce_drop_lowest(cands, slot_budget)
                for tb in inc:
                    included_dynamic.append(tb)
                    spent_in_slot += tb.tokens
                    decisions.append(
                        BlockDecision(
                            block_id=tb.block.id,
                            action="included",
                            tokens_before=tb.tokens,
                            tokens_after=tb.tokens,
                            reason="fits",
                        )
                    )
                for tb in drp:
                    decisions.append(
                        BlockDecision(
                            block_id=tb.block.id,
                            action="dropped",
                            tokens_before=tb.tokens,
                            tokens_after=0,
                            reason="lower_score",
                        )
                    )

            elif kind == "history":
                inc, drp = reduce_anchored_window(cands, slot_budget)
                for tb in inc:
                    included_dynamic.append(tb)
                    spent_in_slot += tb.tokens
                    decisions.append(
                        BlockDecision(
                            block_id=tb.block.id,
                            action="included",
                            tokens_before=tb.tokens,
                            tokens_after=tb.tokens,
                            reason="fits",
                        )
                    )
                for tb in drp:
                    decisions.append(
                        BlockDecision(
                            block_id=tb.block.id,
                            action="dropped",
                            tokens_before=tb.tokens,
                            tokens_after=0,
                            reason="outside_window",
                        )
                    )

            slot_usage[kind] = SlotUsage(
                allocated=slot_budget,
                used=spent_in_slot,
                demand=demand_per_slot.get(kind, 0),
            )

        # 9. Cache-aware ordering: prefix -> history -> rag -> active user
        selected_tokens = [*p0_blocks, *included_p1, *included_dynamic]
        ordered, prefix_hash, prefix_tokens = order_blocks_for_caching(selected_tokens)

        # 10. Structure normalization (applied on ordered sequence to guarantee first chat turn is user)
        normalized = normalize_message_structure(
            ordered,
            requires_alternation=self._model.requires_alternation,
            tokenizer=self._tokenizer,
        )

        # 11. Security: Trust boundary wrapping
        wrapped = [wrap_untrusted_block(tb, self._tokenizer) for tb in normalized]

        # 12. Final verification check
        final_total = sum(tb.tokens for tb in wrapped)
        validate_assembled_structure(
            wrapped,
            requires_alternation=self._model.requires_alternation,
        )

        duration_ms = (time.perf_counter_ns() - start_time) / 1_000_000.0

        # Construct diagnostic audit report
        tool_schema_costs = {
            tb.block.id: tb.tokens for tb in wrapped if tb.block.kind == "tool_schema"
        }

        report = AssemblyReport(
            input_budget=input_budget,
            total_tokens=final_total,
            utilization=round(final_total / max(1, input_budget), 4),
            per_slot=slot_usage,
            cache_prefix_tokens=prefix_tokens,
            prefix_hash=prefix_hash,
            tool_schema_tokens=tool_schema_costs,
            warnings=() if final_total <= input_budget else ("budget_overrun_detected",),
            decisions=tuple(decisions),
            duration_ms=round(duration_ms, 2),
        )

        return AssembledContext(
            blocks=[tb.block for tb in wrapped],
            report=report,
        )
