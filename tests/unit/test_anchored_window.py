"""Unit tests for anchored conversation history windowing."""

from context_assembler.domain.block import (
    ContextBlock,
    Drop,
    Priority,
    TokenizedBlock,
)
from context_assembler.strategies.anchored_window import reduce_anchored_window


def make_turn(msg_id: str, role: str, content: str, tokens: int, group_id: str | None = None) -> TokenizedBlock:
    block = ContextBlock(
        id=msg_id,
        kind="history",
        role=role,  # type: ignore[arg-type]
        content=content,
        priority=Priority.P4,
        strategy=Drop(),
        group_id=group_id,
    )
    return TokenizedBlock(block, tokens)


def test_anchored_window_keeps_turn_zero_and_latest_turns() -> None:
    # 5 conversation turns, 50 tokens each
    turns = [
        make_turn("turn_0_user", "user", "I need to migrate the database", 50),
        make_turn("turn_1_asst", "assistant", "Sure, which db?", 50),
        make_turn("turn_2_user", "user", "Postgres to Spanner", 50),
        make_turn("turn_3_asst", "assistant", "Checking schemas...", 50),
        make_turn("turn_4_user", "user", "What are the table limits?", 50),
    ]

    # Total tokens = 250. Budget = 160 tokens (can fit 3 turns: turn 0 + turn 3 + turn 4)
    included, dropped = reduce_anchored_window(turns, budget=160)

    included_ids = [tb.block.id for tb in included]
    dropped_ids = [tb.block.id for tb in dropped]

    # Anchor turn 0 MUST be kept
    assert "turn_0_user" in included_ids
    # Latest turns preserved
    assert "turn_4_user" in included_ids
    assert "turn_3_asst" in included_ids
    # Middle turn dropped
    assert "turn_1_asst" in dropped_ids or "turn_2_user" in dropped_ids


def test_anchored_window_preserves_atomic_tool_groups() -> None:
    turns = [
        make_turn("t0", "user", "Goal", 30),
        make_turn("t1", "user", "Run check", 30),
        make_turn("t2_call", "assistant", "calling db_status", 40, group_id="grp_1"),
        make_turn("t2_res", "tool", "db is healthy", 40, group_id="grp_1"),
        make_turn("t3", "user", "Final question", 30),
    ]

    # Total is 170. Budget 110.
    # Group grp_1 is 80 tokens together.
    # Anchor t0 is 30.
    included, dropped = reduce_anchored_window(turns, budget=110)
    included_ids = {tb.block.id for tb in included}

    # Either both members of grp_1 are included, or neither is included
    grp_members_included = {"t2_call", "t2_res"}.issubset(included_ids)
    grp_members_excluded = not {"t2_call", "t2_res"}.intersection(included_ids)
    assert grp_members_included or grp_members_excluded
