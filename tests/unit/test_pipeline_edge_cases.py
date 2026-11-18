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
    MiddleOut,
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


def test_single_line_giant_blob_truncates_via_span_fallback() -> None:
    """Single-line minified JSON without newlines must truncate instead of dropping."""
    tokenizer = HeuristicTokenizer(chars_per_token=4.0)
    model = ModelProfile(context_window=600, max_output_tokens=100, safety_margin_ratio=0.0)
    assembler = Assembler(
        model=model,
        tokenizer=tokenizer,
        slots=[
            SlotPolicy("tool_result", min_tokens=100, max_tokens=200, weight=1),
            SlotPolicy("rag", min_tokens=100, max_tokens=200, weight=1),
        ],
    )

    single_line_json = (
        '{"header":"START_MARKER","payload":"' + ("A" * 4000) + '","footer":"END_MARKER"}'
    )
    single_line_prose = " ".join(f"word_{i}" for i in range(600))

    blocks = [
        ContextBlock("h0", "history", "user", "Fetch data", Priority.P0, Never(), seq=0),
        ContextBlock("u_act", "user", "user", "Inspect", Priority.P1, Never()),
        ContextBlock(
            "call_1",
            "tool_call",
            "assistant",
            "Fetching JSON",
            Priority.P2,
            Drop(),
            group_id="g_json",
            seq=1,
        ),
        ContextBlock(
            "res_1",
            "tool_result",
            "tool",
            single_line_json,
            Priority.P2,
            HeadTail(0.5),
            group_id="g_json",
            seq=1,
        ),
        ContextBlock(
            "prose_1",
            "rag",
            "user",
            single_line_prose,
            Priority.P3,
            MiddleOut(),
        ),
    ]

    res = assembler.assemble(blocks)
    by_id = {b.id: b for b in res.blocks}
    assert "res_1" in by_id
    assert "START_MARKER" in by_id["res_1"].content
    assert "END_MARKER" in by_id["res_1"].content
    assert "segments omitted for context budget" in by_id["res_1"].content

    assert "prose_1" in by_id
    assert "word_0" in by_id["prose_1"].content
    assert "word_599" in by_id["prose_1"].content


def test_cross_slot_slack_redistribution_recycles_unused_tokens() -> None:
    """Unused internal fragmentation slack from RAG flows to downstream history slot."""
    tokenizer = HeuristicTokenizer(chars_per_token=1.0)
    model = ModelProfile(
        context_window=1_000,
        max_output_tokens=100,
        per_message_overhead=0,
        safety_margin_ratio=0.0,
    )
    assembler = Assembler(
        model=model,
        tokenizer=tokenizer,
        slots=[
            SlotPolicy("rag", min_tokens=200, max_tokens=500, weight=1),
            SlotPolicy(
                "history", min_tokens=200, max_tokens=700, weight=1, low_watermark_ratio=1.0
            ),
        ],
    )

    # Budget after 50-token P1 user query is 850 tokens.
    # RAG demand is 600 (two 300-token chunks), history demand is 500 (five 100-token turns).
    # Initial water-fill gives ~450 to RAG and ~400 to history.
    # Only one 300-token RAG chunk fits in 450, leaving 150 tokens of slack!
    # Slack recycling transfers +100 tokens to history so all 500 history tokens fit!
    blocks = [
        ContextBlock("u_act", "user", "user", "Q" * 50, Priority.P1, Never()),
        ContextBlock("rag_1", "rag", "user", "R" * 300, Priority.P3, Drop(), score=0.9),
        ContextBlock("rag_2", "rag", "user", "S" * 300, Priority.P3, Drop(), score=0.5),
    ]
    for i in range(5):
        role = "user" if i % 2 == 0 else "assistant"
        blocks.append(
            ContextBlock(f"h_{i}", "history", role, "H" * 100, Priority.P4, Drop(), seq=i)
        )

    res = assembler.assemble(blocks)
    hist_included = [b for b in res.blocks if b.kind == "history"]
    assert len(hist_included) == 5


def test_age_based_tool_observation_masking_and_adapter_tool_metadata() -> None:
    tokenizer = HeuristicTokenizer(chars_per_token=4.0)
    model = ModelProfile(context_window=4_000, max_output_tokens=500)
    assembler = Assembler(
        model=model,
        tokenizer=tokenizer,
        keep_recent_tool_turns=2,
        observation_epoch_stride=2,
    )

    blocks = [
        ContextBlock(
            "sys",
            "system",
            "system",
            "System instructions",
            Priority.P0,
            Never(),
            cacheable=True,
        ),
        ContextBlock("h0", "history", "user", "Start task", Priority.P4, Drop(), seq=0),
        ContextBlock(
            "old_call",
            "tool_call",
            "assistant",
            "Running old query",
            Priority.P2,
            Drop(),
            group_id="g_old",
            tool_name="run_sql",
            tool_input={"query": "SELECT * FROM audit_logs", "limit": 500},
            seq=1,
        ),
        ContextBlock(
            "old_res",
            "tool_result",
            "tool",
            "\n".join(f"row {i}: audit log data entry" for i in range(40)),
            Priority.P2,
            HeadTail(0.5),
            group_id="g_old",
            seq=1,
        ),
        ContextBlock(
            "h4", "history", "user", "Now check current users", Priority.P4, Drop(), seq=5
        ),
        ContextBlock(
            "new_call",
            "tool_call",
            "assistant",
            "Running recent query",
            Priority.P2,
            Drop(),
            group_id="g_new",
            tool_name="run_sql",
            tool_input={"query": "SELECT count(*) FROM users"},
            seq=6,
        ),
        ContextBlock(
            "new_res",
            "tool_result",
            "tool",
            "\n".join(f"recent row {i}: active user record" for i in range(20)),
            Priority.P2,
            HeadTail(0.5),
            group_id="g_new",
            seq=6,
        ),
        ContextBlock("u_act", "user", "user", "Summarize findings", Priority.P1, Never(), seq=7),
    ]

    res = assembler.assemble(blocks)
    by_id = {b.id: b for b in res.blocks}

    # Old observation (seq=1 vs max_seq=7) is masked into a 1-line stub; recent (seq=6) stays full
    assert "Old tool observation from turn 1 masked" in by_id["old_res"].content
    assert "recent row 10: active user record" in by_id["new_res"].content

    # OpenAI adapter preserves tool_name and deterministic JSON arguments
    oai = to_openai_messages(res.blocks)
    call_msg = next(m for m in oai if "tool_calls" in m)
    fn_info = call_msg["tool_calls"][0]["function"]
    assert fn_info["name"] == "run_sql"
    assert fn_info["arguments"] == '{"limit":500,"query":"SELECT * FROM audit_logs"}'

    # Anthropic adapter preserves tool_use input and marks 2nd breakpoint on history
    anth = to_anthropic_payload(res.blocks, enable_cache_control=True, mark_history_breakpoint=True)
    assert anth["system"][0]["cache_control"] == {"type": "ephemeral"}
    hist_msg_with_cache = [
        m
        for m in anth["messages"]
        if isinstance(m["content"], list)
        and any(isinstance(p, dict) and "cache_control" in p for p in m["content"])
    ]
    assert len(hist_msg_with_cache) >= 1
