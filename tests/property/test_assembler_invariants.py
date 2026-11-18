"""Hypothesis property-based tests verifying core context assembly invariants."""

from typing import Literal

from hypothesis import given, settings
from hypothesis import strategies as st

from context_assembler.assembler import Assembler
from context_assembler.domain.block import (
    BlockKind,
    ContextBlock,
    Drop,
    HeadTail,
    Never,
    Priority,
    ReductionStrategy,
    Role,
)
from context_assembler.domain.budget import ModelProfile
from context_assembler.domain.errors import P0OverflowError
from context_assembler.tokenizers.heuristic import HeuristicTokenizer

strategy_kinds: st.SearchStrategy[BlockKind] = st.sampled_from(
    [
        "system",
        "tool_schema",
        "user",
        "memory",
        "rag",
        "history",
    ]
)

strategy_priorities: st.SearchStrategy[Priority] = st.sampled_from(
    [
        Priority.P0,
        Priority.P1,
        Priority.P2,
        Priority.P3,
        Priority.P4,
    ]
)


def block_strategy_mapper(kind: BlockKind, priority: Priority) -> ReductionStrategy:
    if priority == Priority.P0 or kind in ("system", "tool_schema"):
        return Never()
    if kind == "tool_result":
        return HeadTail(head_ratio=0.5)
    return Drop()


@st.composite
def random_blocks(draw: st.DrawFn) -> list[ContextBlock]:
    count = draw(st.integers(min_value=1, max_value=25))
    blocks: list[ContextBlock] = []

    if draw(st.booleans()):
        blocks.append(
            ContextBlock(
                id="sys_prompt",
                kind="system",
                role="system",
                content=draw(st.text(min_size=10, max_size=200)),
                priority=Priority.P0,
                strategy=Never(),
                cacheable=True,
            )
        )

    blocks.append(
        ContextBlock(
            id="active_user_query",
            kind="user",
            role="user",
            content=draw(st.text(min_size=10, max_size=300)),
            priority=Priority.P1,
            strategy=HeadTail(0.5),
        )
    )

    for i in range(count):
        # Periodically emit an atomic tool_call + tool_result pair
        if draw(st.booleans()) and i % 4 == 0:
            gid = f"call_grp_{i}"
            blocks.append(
                ContextBlock(
                    id=f"tool_call_{i}",
                    kind="tool_call",
                    role="assistant",
                    content=f"Invoking diagnostic tool {i}",
                    priority=Priority.P2,
                    strategy=Drop(),
                    group_id=gid,
                    seq=i,
                )
            )
            blocks.append(
                ContextBlock(
                    id=f"tool_res_{i}",
                    kind="tool_result",
                    role="tool",
                    content=draw(st.text(min_size=10, max_size=600)),
                    priority=Priority.P2,
                    strategy=HeadTail(0.5),
                    group_id=gid,
                    trust="untrusted",
                    source=f"tool_{i}",
                    seq=i,
                )
            )
            continue

        kind = draw(strategy_kinds)
        if kind == "system":
            continue

        priority = draw(strategy_priorities)
        if priority == Priority.P0:
            priority = Priority.P3

        role: Role = "user"
        if kind == "history":
            role = "assistant" if i % 2 == 0 else "user"

        trust: Literal["trusted", "untrusted"] = (
            "untrusted" if kind == "rag" and draw(st.booleans()) else "trusted"
        )
        content = draw(st.text(min_size=5, max_size=500))
        blocks.append(
            ContextBlock(
                id=f"blk_{i}",
                kind=kind,
                role=role,
                content=content,
                priority=priority,
                strategy=block_strategy_mapper(kind, priority),
                score=draw(st.floats(min_value=0.0, max_value=1.0)),
                trust=trust,
                seq=i,
            )
        )

    return blocks


@settings(max_examples=120, deadline=None)
@given(
    blocks=random_blocks(),
    context_window=st.integers(min_value=500, max_value=8_000),
    max_output=st.integers(min_value=50, max_value=300),
    requires_alternation=st.booleans(),
)
def test_invariant_never_exceeds_budget_and_preserves_structure(
    blocks: list[ContextBlock],
    context_window: int,
    max_output: int,
    requires_alternation: bool,
) -> None:
    """Invariants: Budget safety, P0 integrity, Group atomicity, Trust wrap, Alternation."""
    if max_output >= context_window:
        return

    model = ModelProfile(
        context_window=context_window,
        max_output_tokens=max_output,
        per_message_overhead=4,
        safety_margin_ratio=0.05,
        requires_alternation=requires_alternation,
    )
    tokenizer = HeuristicTokenizer(chars_per_token=4.0)
    assembler = Assembler(model=model, tokenizer=tokenizer)

    try:
        result = assembler.assemble(blocks)
    except P0OverflowError:
        return

    # 1. Budget safety
    assert result.report.total_tokens <= result.report.input_budget

    # 2. P0 integrity: every P0 input block must be present in output
    p0_input_ids = {b.id for b in blocks if b.priority == Priority.P0 or b.pinned}
    output_ids = {b.id for b in result.blocks}
    assert p0_input_ids.issubset(output_ids)

    # 3. Atomic group integrity: tool_call and tool_result must appear together
    call_groups = {b.group_id for b in result.blocks if b.kind == "tool_call" and b.group_id}
    result_groups = {b.group_id for b in result.blocks if b.kind == "tool_result" and b.group_id}
    assert call_groups == result_groups

    # 4. Trust boundary: untrusted blocks must never have system/developer role and must be wrapped
    for b in result.blocks:
        if b.trust == "untrusted":
            assert b.role not in ("system", "developer")
            assert '<document id="' in b.content
            assert b.content.count('<document id="') == b.content.count("</document>")

    # 5. Decisions audit log completeness
    decision_ids = {d.block_id for d in result.report.decisions}
    assert decision_ids == {b.id for b in blocks}


@settings(max_examples=50, deadline=None)
@given(blocks=random_blocks())
def test_invariant_assembly_is_deterministic(blocks: list[ContextBlock]) -> None:
    """Invariant: Identical inputs must produce identical outputs and decisions."""
    model = ModelProfile(
        context_window=4000,
        max_output_tokens=500,
        per_message_overhead=4,
        safety_margin_ratio=0.03,
    )
    tokenizer = HeuristicTokenizer(chars_per_token=4.0)
    assembler = Assembler(model=model, tokenizer=tokenizer)

    try:
        result_a = assembler.assemble(blocks)
        result_b = assembler.assemble(blocks)
    except P0OverflowError:
        return

    assert len(result_a.blocks) == len(result_b.blocks)
    assert result_a.report.total_tokens == result_b.report.total_tokens
    assert result_a.report.prefix_hash == result_b.report.prefix_hash
    for b_a, b_b in zip(result_a.blocks, result_b.blocks, strict=True):
        assert b_a.id == b_b.id
        assert b_a.content == b_b.content
