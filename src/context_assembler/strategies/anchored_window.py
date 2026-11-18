"""Anchored window reduction for conversation histories."""

from __future__ import annotations

from collections.abc import Sequence

from context_assembler.domain.block import TokenizedBlock


def reduce_anchored_window(
    history_blocks: Sequence[TokenizedBlock],
    budget: int,
) -> tuple[list[TokenizedBlock], list[TokenizedBlock]]:
    """Preserves initial goal anchor (first turn) + latest turns while dropping middle.

    Respects atomic group boundaries so tool calls and results are evicted or preserved
    together as a cohesive unit.
    """
    total_tokens = sum(tb.tokens for tb in history_blocks)
    if total_tokens <= budget:
        return list(history_blocks), []

    if not history_blocks or budget <= 0:
        return [], list(history_blocks)

    # Turn 0 (anchor) is the first message or first atomic group
    anchor_candidates: list[TokenizedBlock] = []
    first_block = history_blocks[0]
    if first_block.block.group_id:
        target_group = first_block.block.group_id
        anchor_candidates = [tb for tb in history_blocks if tb.block.group_id == target_group]
    else:
        anchor_candidates = [first_block]

    anchor_cost = sum(tb.tokens for tb in anchor_candidates)
    if anchor_cost > budget:
        # Not even the anchor fits in the allocated history budget
        return [], list(history_blocks)

    included_set: set[str] = {tb.block.id for tb in anchor_candidates}
    spent = anchor_cost

    # Iterate from the end (newest messages first) and include as many as fit
    rest = [tb for tb in history_blocks if tb.block.id not in included_set]
    reversed_rest = list(reversed(rest))

    i = 0
    while i < len(reversed_rest):
        current = reversed_rest[i]

        # If part of an atomic group, bundle all members of that group
        if current.block.group_id:
            group_id = current.block.group_id
            group_members = [tb for tb in reversed_rest if tb.block.group_id == group_id]
            group_cost = sum(tb.tokens for tb in group_members)

            if spent + group_cost <= budget:
                for gm in group_members:
                    included_set.add(gm.block.id)
                spent += group_cost
            # Skip past other group members in the loop
            i += len(group_members)
        else:
            if spent + current.tokens <= budget:
                included_set.add(current.block.id)
                spent += current.tokens
            i += 1

    # Preserve original chronological sequence order
    included = [tb for tb in history_blocks if tb.block.id in included_set]
    dropped = [tb for tb in history_blocks if tb.block.id not in included_set]

    return included, dropped
