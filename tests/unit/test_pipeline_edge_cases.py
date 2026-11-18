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
