"""Reproducible evaluation & ablation benchmark for Context Assembler Engine.

Compares:
- Baseline A: Naive Tail Truncate (character/token clip from the end)
- Baseline B: Sliding Window (keep system + newest N messages that fit)
- Engine C:   Context Assembler v0.2 (water-fill + HeadTail + AnchoredWindow + hysteresis)
- Ablations:  No-HeadTail, No-Anchor, No-Hysteresis

Run with:
    .venv/bin/python eval/run_eval.py
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from context_assembler import (
    Assembler,
    ContextBlock,
    Drop,
    HeadTail,
    HeuristicTokenizer,
    ModelProfile,
    Never,
    Priority,
    SlotPolicy,
)


@dataclass(frozen=True)
class EvalScenario:
    """Single evaluation scenario with ground-truth information needles."""

    name: str
    suite: str
    blocks: list[ContextBlock]
    required_needles: list[str]


def build_scenarios() -> list[EvalScenario]:
    """Constructs 45 deterministic evaluation scenarios across 3 suites."""
    scenarios: list[EvalScenario] = []

    # Suite 1: Long Multi-Turn Goal Retention (15 scenarios)
    for i in range(15):
        constraint = f"CONSTRAINT_TENANT_ISO_{i:02d}: strictly filter by tenant_id='org_{i}'"
        active_q = f"QUERY_FINAL_{i:02d}: generate the SQL migration now."
        blocks: list[ContextBlock] = [
            ContextBlock(
                id="sys",
                kind="system",
                role="system",
                content="You are a production database migration assistant.",
                priority=Priority.P0,
                strategy=Never(),
                cacheable=True,
            ),
            ContextBlock(
                id="h0_anchor",
                kind="history",
                role="user",
                content=f"Initial goal: {constraint} across all tables.",
                priority=Priority.P4,
                strategy=Drop(),
                seq=0,
            ),
        ]
        for turn in range(1, 25):
            role = "assistant" if turn % 2 == 1 else "user"
            blocks.append(
                ContextBlock(
                    id=f"h_{turn}",
                    kind="history",
                    role=role,
                    content=(
                        f"Discussion turn {turn} covering schema columns, "
                        f"intermediate index notes, and staging checks ({'x' * 120})."
                    ),
                    priority=Priority.P4,
                    strategy=Drop(),
                    seq=turn,
                )
            )
        blocks.append(
            ContextBlock(
                id="active_user",
                kind="user",
                role="user",
                content=active_q,
                priority=Priority.P1,
                strategy=HeadTail(0.5),
            )
        )
        scenarios.append(
            EvalScenario(
                name=f"multi_turn_{i:02d}",
                suite="multi_turn_goal",
                blocks=blocks,
                required_needles=[constraint, active_q],
            )
        )

    # Suite 2: Tool-Heavy Log Needle Recall (15 scenarios)
    for i in range(15):
        head_needle = f"CMD_EXEC_HEADER_{i:02d}: pg_restore --clean --dbname=prod_{i}"
        tail_needle = f"FATAL_ERR_FOOTER_{i:02d}: deadlock detected on relation orders_{i}"
        log_body = [head_needle]
        for line_idx in range(120):
            offset = line_idx * 1024
            log_body.append(
                f"  [trace {line_idx:03d}] processing batch block offset={offset} status=OK"
            )
        log_body.append(tail_needle)

        blocks = [
            ContextBlock(
                id="sys",
                kind="system",
                role="system",
                content="You are an SRE incident responder.",
                priority=Priority.P0,
                strategy=Never(),
                cacheable=True,
            ),
            ContextBlock(
                id="h0",
                kind="history",
                role="user",
                content="Restore job failed in production.",
                priority=Priority.P4,
                strategy=Drop(),
                seq=0,
            ),
            ContextBlock(
                id="call_restore",
                kind="tool_call",
                role="assistant",
                content="Inspecting restore log output...",
                priority=Priority.P2,
                strategy=Drop(),
                group_id=f"grp_tool_{i}",
                seq=1,
            ),
            ContextBlock(
                id="res_restore",
                kind="tool_result",
                role="tool",
                content="\n".join(log_body),
                priority=Priority.P2,
                strategy=HeadTail(0.5),
                group_id=f"grp_tool_{i}",
                trust="untrusted",
                source="bash:pg_restore",
                seq=1,
            ),
            ContextBlock(
                id="active_user",
                kind="user",
                role="user",
                content="What command was run and what fatal error terminated it?",
                priority=Priority.P1,
                strategy=HeadTail(0.5),
            ),
        ]
        scenarios.append(
            EvalScenario(
                name=f"tool_heavy_{i:02d}",
                suite="tool_log_needle",
                blocks=blocks,
                required_needles=[head_needle, tail_needle],
            )
        )

    # Suite 3: Distractor-Heavy RAG Grounding (15 scenarios)
    for i in range(15):
        gold_fact_1 = f"GOLD_FACT_A_{i:02d}: max_connections must be set to 250 behind PgBouncer."
        gold_fact_2 = f"GOLD_FACT_B_{i:02d}: statement_timeout must be 15000ms for web workers."
        blocks = [
            ContextBlock(
                id="sys",
                kind="system",
                role="system",
                content="Answer strictly using retrieved KB documents.",
                priority=Priority.P0,
                strategy=Never(),
                cacheable=True,
            ),
        ]
        # Add 18 low-score distractor chunks first
        for d_idx in range(18):
            blocks.append(
                ContextBlock(
                    id=f"rag_distractor_{d_idx:02d}",
                    kind="rag",
                    role="user",
                    content=(
                        f"Distractor doc {d_idx}: general background notes on legacy "
                        f"replication and archival tape storage ({'y' * 160})."
                    ),
                    priority=Priority.P3,
                    strategy=Drop(),
                    score=0.10 + (d_idx * 0.02),
                    trust="untrusted",
                )
            )
        # Add 2 high-score gold chunks
        blocks.append(
            ContextBlock(
                id="rag_gold_1",
                kind="rag",
                role="user",
                content=f"Production runbook section 4.1: {gold_fact_1}",
                priority=Priority.P3,
                strategy=Drop(),
                score=0.96,
                trust="untrusted",
            )
        )
        blocks.append(
            ContextBlock(
                id="rag_gold_2",
                kind="rag",
                role="user",
                content=f"Production runbook section 4.2: {gold_fact_2}",
                priority=Priority.P3,
                strategy=Drop(),
                score=0.92,
                trust="untrusted",
            )
        )
        blocks.append(
            ContextBlock(
                id="active_user",
                kind="user",
                role="user",
                content="What are the exact max_connections and statement_timeout values?",
                priority=Priority.P1,
                strategy=HeadTail(0.5),
            )
        )
        scenarios.append(
            EvalScenario(
                name=f"rag_grounding_{i:02d}",
                suite="rag_grounding",
                blocks=blocks,
                required_needles=[gold_fact_1, gold_fact_2],
            )
        )

    return scenarios


def check_protocol_valid(blocks: list[ContextBlock]) -> bool:
    """Checks if assembled blocks satisfy provider tool-pairing and role rules."""
    chat = [b for b in blocks if b.role not in ("system", "developer")]
    if not chat or chat[0].role in ("assistant", "tool"):
        return False

    seen_calls: set[str] = set()
    seen_results: set[str] = set()
    for b in chat:
        if b.kind == "tool_call" and b.group_id:
            seen_calls.add(b.group_id)
        elif b.kind == "tool_result" and b.group_id:
            if b.group_id not in seen_calls:
                return False
            seen_results.add(b.group_id)

    return seen_calls == seen_results


def run_naive_truncate(
    blocks: list[ContextBlock],
    input_budget: int,
    tokenizer: HeuristicTokenizer,
) -> list[ContextBlock]:
    """Baseline A: Concatenate blocks in input order and hard-truncate when budget fills."""
    out: list[ContextBlock] = []
    spent = 0
    for b in blocks:
        cost = tokenizer.count(b.content)
        if spent + cost <= input_budget:
            out.append(b)
            spent += cost
        else:
            rem_chars = max(0, (input_budget - spent) * 4)
            if rem_chars > 20:
                clipped = b.content[:rem_chars]
                out.append(
                    ContextBlock(
                        id=b.id,
                        kind=b.kind,
                        role=b.role,
                        content=clipped,
                        priority=b.priority,
                        strategy=b.strategy,
                        group_id=b.group_id,
                    )
                )
            break
    return out


def run_sliding_window(
    blocks: list[ContextBlock],
    input_budget: int,
    tokenizer: HeuristicTokenizer,
) -> list[ContextBlock]:
    """Baseline B: Keep system prompt + latest N blocks from the end that fit whole."""
    sys_blocks = [b for b in blocks if b.role in ("system", "developer")]
    rest = [b for b in blocks if b.role not in ("system", "developer")]

    spent = sum(tokenizer.count(b.content) for b in sys_blocks)
    kept_rev: list[ContextBlock] = []
    for b in reversed(rest):
        cost = tokenizer.count(b.content)
        if spent + cost <= input_budget:
            kept_rev.append(b)
            spent += cost
    return [*sys_blocks, *reversed(kept_rev)]


def bootstrap_ci(values: list[float], seed: int = 42) -> tuple[float, float]:
    """Computes 95% bootstrap confidence interval for the mean."""
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(1000):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return (round(means[25] * 100, 1), round(means[975] * 100, 1))


def simulate_multiturn_cache_cost() -> dict[str, dict[str, float]]:
    """Simulates a 20-turn coding agent session to measure KV prefix cache reuse."""
    tokenizer = HeuristicTokenizer(chars_per_token=4.0)
    model = ModelProfile(
        context_window=1_200,
        max_output_tokens=200,
        per_message_overhead=0,
        safety_margin_ratio=0.0,
    )

    asm_hyst = Assembler(
        model=model,
        tokenizer=tokenizer,
        slots=[
            SlotPolicy(
                "history", min_tokens=200, max_tokens=800, weight=1, low_watermark_ratio=0.65
            )
        ],
    )
    asm_no_hyst = Assembler(
        model=model,
        tokenizer=tokenizer,
        slots=[
            SlotPolicy("history", min_tokens=200, max_tokens=800, weight=1, low_watermark_ratio=1.0)
        ],
    )

    sys_block = ContextBlock(
        id="sys",
        kind="system",
        role="system",
        content="S" * 600,
        priority=Priority.P0,
        strategy=Never(),
        cacheable=True,
    )

    history: list[ContextBlock] = []
    results: dict[str, dict[str, float]] = {}

    for label, mode in [
        ("sliding_window", "sliding"),
        ("assembler_no_hysteresis", "no_hyst"),
        ("assembler_v0.2", "hyst"),
    ]:
        history.clear()
        prev_tokens_list: list[str] = []
        total_prompt_tokens = 0
        total_cached_tokens = 0

        for turn in range(20):
            role = "user" if turn % 2 == 0 else "assistant"
            history.append(
                ContextBlock(
                    id=f"h_{turn}",
                    kind="history",
                    role=role,
                    content=f"Turn_{turn}_" + ("T" * 240),
                    priority=Priority.P4,
                    strategy=Drop(),
                    seq=turn,
                )
            )
            active = ContextBlock(
                id="u_act",
                kind="user",
                role="user",
                content=f"Active_{turn}_" + ("Q" * 100),
                priority=Priority.P1,
                strategy=HeadTail(0.5),
            )
            candidates = [sys_block, *history, active]

            if mode == "sliding":
                out_blocks = run_sliding_window(candidates, 1_000, tokenizer)
            elif mode == "no_hyst":
                out_blocks = asm_no_hyst.assemble(candidates).blocks
            else:
                out_blocks = asm_hyst.assemble(candidates).blocks

            curr_segments = [f"{b.role}:{b.content}" for b in out_blocks]
            # Longest common prefix of identical message blocks hits KV cache
            cached = 0
            for prev_seg, curr_seg in zip(prev_tokens_list, curr_segments, strict=False):
                if prev_seg == curr_seg:
                    cached += tokenizer.count(curr_seg.split(":", 1)[1])
                else:
                    break

            turn_total = sum(tokenizer.count(b.content) for b in out_blocks)
            total_prompt_tokens += turn_total
            total_cached_tokens += cached
            prev_tokens_list = curr_segments

        hit_ratio = total_cached_tokens / max(1, total_prompt_tokens)
        # Cost per 1K turns at $3.00/1M uncached, $0.30/1M cached (90% discount)
        uncached_t = total_prompt_tokens - total_cached_tokens
        cost_20_turns = (uncached_t * 3.0 + total_cached_tokens * 0.30) / 1_000_000
        cost_per_1k_turns = cost_20_turns * (1000 / 20)

        results[label] = {
            "cache_hit_pct": round(hit_ratio * 100, 1),
            "cost_per_1k_turns_usd": round(cost_per_1k_turns, 4),
        }

    return results


def main() -> None:
    tokenizer = HeuristicTokenizer(chars_per_token=4.0)
    scenarios = build_scenarios()
    model = ModelProfile(
        context_window=1_600,
        max_output_tokens=400,
        per_message_overhead=4,
        safety_margin_ratio=0.03,
    )
    assembler = Assembler(
        model=model,
        tokenizer=tokenizer,
        slots=[
            SlotPolicy("tool_result", min_tokens=300, max_tokens=700, weight=3),
            SlotPolicy("rag", min_tokens=250, max_tokens=700, weight=4),
            SlotPolicy(
                "history", min_tokens=200, max_tokens=600, weight=2, low_watermark_ratio=0.70
            ),
        ],
    )
    input_budget = 1_100

    methods = [
        "naive_truncate",
        "sliding_window",
        "assembler_no_headtail",
        "context_assembler_v0.2",
    ]
    recalls: dict[str, list[float]] = {m: [] for m in methods}
    validities: dict[str, list[float]] = {m: [] for m in methods}

    for sc in scenarios:
        for m in methods:
            if m == "naive_truncate":
                out_blocks = run_naive_truncate(sc.blocks, input_budget, tokenizer)
            elif m == "sliding_window":
                out_blocks = run_sliding_window(sc.blocks, input_budget, tokenizer)
            elif m == "assembler_no_headtail":
                no_ht_blocks = [
                    ContextBlock(
                        id=b.id,
                        kind=b.kind,
                        role=b.role,
                        content=b.content,
                        priority=b.priority,
                        strategy=Drop() if isinstance(b.strategy, HeadTail) else b.strategy,
                        score=b.score,
                        group_id=b.group_id,
                        cacheable=b.cacheable,
                        seq=b.seq,
                        trust=b.trust,
                        source=b.source,
                    )
                    for b in sc.blocks
                ]
                out_blocks = assembler.assemble(no_ht_blocks).blocks
            else:
                out_blocks = assembler.assemble(sc.blocks).blocks

            joined = "\n".join(b.content for b in out_blocks)
            found = sum(1 for needle in sc.required_needles if needle in joined)
            recalls[m].append(found / len(sc.required_needles))
            validities[m].append(1.0 if check_protocol_valid(out_blocks) else 0.0)

    cache_sim = simulate_multiturn_cache_cost()

    summary: dict[str, Any] = {"benchmark_suites": {}, "cache_simulation": cache_sim}
    print("=" * 84)
    print("📊 CONTEXT ASSEMBLER EVALUATION & ABLATION BENCHMARK (45 Scenarios)")
    print("=" * 84)
    print(
        f"{'Method':26s} | {'Needle Recall (%)':22s} | {'95% Bootstrap CI':18s} | {'API Valid (%)'}"
    )
    print("-" * 84)

    for m in methods:
        mean_rec = round(sum(recalls[m]) / len(recalls[m]) * 100, 1)
        ci_low, ci_high = bootstrap_ci(recalls[m])
        mean_val = round(sum(validities[m]) / len(validities[m]) * 100, 1)
        summary["benchmark_suites"][m] = {
            "needle_recall_pct": mean_rec,
            "ci_95": [ci_low, ci_high],
            "protocol_valid_pct": mean_val,
        }
        ci_str = f"[{ci_low:4.1f}%, {ci_high:4.1f}%]"
        print(f"{m:26s} | {mean_rec:17.1f}%     | {ci_str:18s} | {mean_val:10.1f}%")

    print("\n" + "=" * 84)
    print("⚡ 20-TURN CODING AGENT PROMPT CACHE & COST SIMULATION")
    print("=" * 84)
    for k, v in cache_sim.items():
        print(
            f"  • {k:26s} -> KV Cache Hit Rate: {v['cache_hit_pct']:5.1f}% | "
            f"Cost / 1K Turns: ${v['cost_per_1k_turns_usd']:.4f}"
        )
    print("=" * 84)

    out_path = Path(__file__).parent / "results.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
