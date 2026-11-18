"""Verification checks against provider API invariant rules."""

from __future__ import annotations

from collections.abc import Sequence

from context_assembler.domain.block import TokenizedBlock
from context_assembler.domain.errors import StructuralValidityError


def validate_assembled_structure(
    blocks: Sequence[TokenizedBlock],
    requires_alternation: bool = False,
) -> None:
    """Verifies that an assembled block sequence is guaranteed valid for provider dispatch."""
    if not blocks:
        return

    chat_blocks = [tb for tb in blocks if tb.block.role not in ("system", "developer")]
    if not chat_blocks:
        return

    # Invariant 1: First chat turn must be a user message
    if chat_blocks[0].block.role in ("assistant", "tool"):
        raise StructuralValidityError(
            rule="first_chat_turn_must_be_user",
            detail=f"Chat history begins with {chat_blocks[0].block.role!r} instead of 'user'.",
        )

    # Invariant 2: Atomic tool call/result pairing in final assembled order
    seen_calls: set[str] = set()
    seen_results: set[str] = set()
    for tb in chat_blocks:
        gid = tb.block.group_id
        if tb.block.kind == "tool_call" and gid:
            seen_calls.add(gid)
        elif tb.block.kind == "tool_result" and gid:
            if gid not in seen_calls:
                raise StructuralValidityError(
                    rule="tool_result_must_follow_tool_call",
                    detail=f"Tool result in group {gid!r} appears without a preceding tool call.",
                )
            seen_results.add(gid)

    unmatched_calls = seen_calls - seen_results
    if unmatched_calls:
        missing = sorted(unmatched_calls)
        raise StructuralValidityError(
            rule="tool_call_must_have_tool_result",
            detail=f"Tool call groups {missing!r} have no corresponding tool result.",
        )

    # Invariant 3: Consecutive roles if strict alternation is mandated
    if requires_alternation:
        for i in range(len(chat_blocks) - 1):
            curr_b = chat_blocks[i].block
            next_b = chat_blocks[i + 1].block
            if (
                curr_b.role == next_b.role
                and curr_b.role in ("user", "assistant")
                and not (curr_b.group_id or next_b.group_id)
            ):
                raise StructuralValidityError(
                    rule="strict_alternation_required",
                    detail=(
                        f"Consecutive messages found with role {curr_b.role!r} "
                        f"at indices {i} and {i + 1}"
                    ),
                )
