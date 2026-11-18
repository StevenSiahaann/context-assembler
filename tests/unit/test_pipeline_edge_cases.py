"""Unit tests for edge cases: atomic tool group eviction, alternatives, dedup, supersession."""

import pytest

from context_assembler.adapters.anthropic import to_anthropic_payload
from context_assembler.adapters.openai import to_openai_messages
from context_assembler.assembler import Assembler
from context_assembler.domain.block import (
    Alternatives,
    ContextBlock,
    Drop,
    HeadTail,
    Never,
    Priority,
)
from context_assembler.domain.budget import ModelProfile, SlotPolicy
from context_assembler.domain.errors import OrphanToolResultError
from context_assembler.tokenizers.heuristic import HeuristicTokenizer


def test_atomic_tool_group_retains_both_call_and_result() -> None:
    model = ModelProfile(context_window=2_000, max_output_tokens=200)
    tokenizer = HeuristicTokenizer(chars_per_token=4.0)
    assembler = Assembler(model=model, tokenizer=tokenizer)

    blocks = [
        ContextBlock("u0", "history", "user", "Check database status", Priority.P4, Drop(), seq=0),
        ContextBlock(
            "call_1",
            "tool_call",
            "assistant",
            "Running pg_isready",
            Priority.P2,
            Drop(),
            group_id="grp_pg",
            source="pg_isready",
            seq=1,
        ),
        ContextBlock(
            "res_1",
            "tool_result",
            "tool",
            "/var/run/postgresql:5432 - accepting connections",
            Priority.P2,
            HeadTail(0.5),
            group_id="grp_pg",
            seq=1,
        ),
        ContextBlock("u_active", "user", "user", "Is it ready?", Priority.P1, Never()),
    ]

    res = assembler.assemble(blocks)
    ids = [b.id for b in res.blocks]
    assert "call_1" in ids
    assert "res_1" in ids
    assert ids.index("call_1") < ids.index("res_1")

    # Verify OpenAI & Anthropic adapters emit tool_calls / tool_use
    oai = to_openai_messages(res.blocks)
    assert any("tool_calls" in m for m in oai)
    anth = to_anthropic_payload(res.blocks)
    assert any(
        isinstance(m["content"], list) and m["content"][-1]["type"] == "tool_use"
        for m in anth["messages"]
    )


def test_atomic_tool_group_drops_together_when_budget_exhausted() -> None:
    model = ModelProfile(context_window=300, max_output_tokens=100)
    tokenizer = HeuristicTokenizer(chars_per_token=1.0)
    assembler = Assembler(
        model=model,
        tokenizer=tokenizer,
        slots=[SlotPolicy("tool_result", min_tokens=0, max_tokens=20, weight=1)],
    )

    blocks = [
        ContextBlock("u_active", "user", "user", "Hi", Priority.P1, Never()),
        ContextBlock(
            "call_1",
            "tool_call",
            "assistant",
            "A" * 50,
            Priority.P2,
            Drop(),
            group_id="grp_big",
        ),
        ContextBlock(
            "res_1",
            "tool_result",
            "tool",
            "B" * 50,
            Priority.P2,
            Drop(),
            group_id="grp_big",
        ),
    ]

    res = assembler.assemble(blocks)
    ids = {b.id for b in res.blocks}
    assert "call_1" not in ids
    assert "res_1" not in ids
    reasons = {d.block_id: d.reason for d in res.report.decisions}
    assert reasons["call_1"] == "group_dropped"
    assert reasons["res_1"] == "group_dropped"


def test_orphan_tool_call_or_result_raises_error() -> None:
    model = ModelProfile(context_window=1_000, max_output_tokens=100)
    assembler = Assembler(model=model, tokenizer=HeuristicTokenizer())

    orphan_call = [
        ContextBlock("u1", "user", "user", "Hi", Priority.P1, Never()),
        ContextBlock(
            "call_only",
            "tool_call",
            "assistant",
            "Calling tool",
            Priority.P2,
            Drop(),
            group_id="g1",
        ),
    ]
    with pytest.raises(OrphanToolResultError):
        assembler.assemble(orphan_call)


def test_alternatives_strategy_picks_first_fitting_variant() -> None:
    model = ModelProfile(context_window=250, max_output_tokens=100, safety_margin_ratio=0.0)
    tokenizer = HeuristicTokenizer(chars_per_token=1.0)
    assembler = Assembler(
        model=model,
        tokenizer=tokenizer,
        slots=[SlotPolicy("rag", min_tokens=10, max_tokens=60, weight=1)],
    )

    blocks = [
        ContextBlock("u1", "user", "user", "Summary please", Priority.P1, Never()),
        ContextBlock(
            id="doc_alt",
            kind="rag",
            role="user",
            content="X" * 200,
            priority=Priority.P3,
            strategy=Alternatives(variants=("X" * 120, "Short summary fits", "Tiny")),
        ),
    ]

    res = assembler.assemble(blocks)
    doc = next(b for b in res.blocks if b.id == "doc_alt")
    assert doc.content == "Short summary fits"


def test_dedup_and_supersession_preserve_history_repeats() -> None:
    model = ModelProfile(context_window=2_000, max_output_tokens=200)
    assembler = Assembler(model=model, tokenizer=HeuristicTokenizer())

    blocks = [
        # Repeated history phrase "OK" in different turns must NOT be deduplicated
        ContextBlock("h0", "history", "user", "OK", Priority.P4, Drop(), seq=0),
        ContextBlock("h1", "history", "assistant", "Proceeding", Priority.P4, Drop(), seq=1),
        ContextBlock("h2", "history", "user", "OK", Priority.P4, Drop(), seq=2),
        # Duplicate RAG content: only higher score kept
        ContextBlock("rag_low", "rag", "user", "Same fact", Priority.P3, Drop(), score=0.2),
        ContextBlock("rag_high", "rag", "user", "Same fact", Priority.P3, Drop(), score=0.9),
        # Superseded file reads: older read replaced with stub
        ContextBlock(
            "read_old",
            "memory",
            "user",
            "old file content v1",
            Priority.P3,
            Drop(),
            supersedes_key="file:app.py",
            seq=1,
        ),
        ContextBlock(
            "read_new",
            "memory",
            "user",
            "new file content v2",
            Priority.P3,
            Drop(),
            supersedes_key="file:app.py",
            seq=5,
        ),
        ContextBlock("u_active", "user", "user", "Done?", Priority.P1, Never()),
    ]

    res = assembler.assemble(blocks)
    by_id = {b.id: b for b in res.blocks}

    assert "h0" in by_id and "h2" in by_id
    assert "rag_high" in by_id and "rag_low" not in by_id
    assert "superseded by later read" in by_id["read_old"].content
    assert by_id["read_new"].content == "new file content v2"


def test_hysteresis_epoch_eviction_stabilizes_history_prefix_across_turns() -> None:
    """Epoch-chunked eviction keeps the first post-anchor turn stable across multiple turns."""
    tokenizer = HeuristicTokenizer(chars_per_token=1.0)
    model = ModelProfile(
        context_window=600,
        max_output_tokens=100,
        per_message_overhead=0,
        safety_margin_ratio=0.0,
    )

    # Compare 1-by-1 eviction (low_watermark_ratio=1.0) vs epoch hysteresis (0.65)
    asm_no_hyst = Assembler(
        model=model,
        tokenizer=tokenizer,
        slots=[
            SlotPolicy("history", min_tokens=100, max_tokens=200, weight=1, low_watermark_ratio=1.0)
        ],
    )
    asm_hyst = Assembler(
        model=model,
        tokenizer=tokenizer,
        slots=[
            SlotPolicy(
                "history", min_tokens=100, max_tokens=200, weight=1, low_watermark_ratio=0.65
            )
        ],
    )

    history: list[ContextBlock] = []
    shifts_no_hyst = 0
    shifts_hyst = 0
    prev_first_post_anchor_no_hyst: str | None = None
    prev_first_post_anchor_hyst: str | None = None

    for turn_idx in range(16):
        role = "user" if turn_idx % 2 == 0 else "assistant"
        history.append(
            ContextBlock(
                id=f"turn_{turn_idx}",
                kind="history",
                role=role,
                content="M" * 20,
                priority=Priority.P4,
                strategy=Drop(),
                seq=turn_idx,
            )
        )
        active = ContextBlock("active_q", "user", "user", "Q" * 10, Priority.P1, Never())

        res_no = asm_no_hyst.assemble([*history, active])
        res_hy = asm_hyst.assemble([*history, active])

        hist_no = [b.id for b in res_no.blocks if b.kind == "history"]
        hist_hy = [b.id for b in res_hy.blocks if b.kind == "history"]

        if len(hist_no) > 1:
            if (
                prev_first_post_anchor_no_hyst is not None
                and hist_no[1] != prev_first_post_anchor_no_hyst
            ):
                shifts_no_hyst += 1
            prev_first_post_anchor_no_hyst = hist_no[1]

        if len(hist_hy) > 1:
            if (
                prev_first_post_anchor_hyst is not None
                and hist_hy[1] != prev_first_post_anchor_hyst
            ):
                shifts_hyst += 1
            prev_first_post_anchor_hyst = hist_hy[1]

    # Hysteresis must cause strictly fewer prefix boundary shifts than 1-by-1 eviction
    assert shifts_hyst < shifts_no_hyst


def test_two_phase_plan_api_for_budget_aware_retrieval() -> None:
    tokenizer = HeuristicTokenizer(chars_per_token=1.0)
    model = ModelProfile(
        context_window=2_000,
        max_output_tokens=400,
        per_message_overhead=4,
        safety_margin_ratio=0.05,
    )
    assembler = Assembler(
        model=model,
        tokenizer=tokenizer,
        slots=[
            SlotPolicy("rag", min_tokens=200, max_tokens=800, weight=3),
            SlotPolicy("history", min_tokens=100, max_tokens=600, weight=1),
        ],
    )

    fixed = [
        ContextBlock("sys", "system", "system", "S" * 100, Priority.P0, Never()),
        ContextBlock("usr", "user", "user", "U" * 50, Priority.P1, Never()),
    ]

    budget_plan = assembler.plan(fixed, demand={"history": 200}, estimated_messages=10)
    assert budget_plan.p0_tokens == 100
    assert budget_plan.p1_tokens == 50
    assert budget_plan.slot_allocations["rag"] > 0

    top_k = budget_plan.recommended_top_k("rag", avg_chunk_tokens=100)
    assert top_k == budget_plan.slot_allocations["rag"] // 100

    # Fetch only top_k distinct chunks and assemble using the pre-computed plan
    rag_chunks = [
        ContextBlock(
            id=f"chunk_{i}",
            kind="rag",
            role="user",
            content=f"Chunk_{i:02d}_" + ("C" * 80),
            priority=Priority.P3,
            strategy=Drop(),
            score=1.0 - i * 0.1,
        )
        for i in range(top_k)
    ]
    res = assembler.assemble([*fixed, *rag_chunks], plan=budget_plan)
    assert res.report.total_tokens <= res.report.input_budget
    included_rag = [b for b in res.blocks if b.kind == "rag"]
    assert len(included_rag) == top_k
