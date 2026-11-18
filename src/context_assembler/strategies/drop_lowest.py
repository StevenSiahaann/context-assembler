"""Drop-lowest-score reduction strategy for competitive document collections."""

from __future__ import annotations

from collections.abc import Sequence

from context_assembler.domain.block import TokenizedBlock


def reduce_drop_lowest(
    blocks: Sequence[TokenizedBlock],
    budget: int,
) -> tuple[list[TokenizedBlock], list[TokenizedBlock]]:
    """Filters blocks to fit within budget, dropping lowest ranked blocks first.

    Ranking prioritizes:
    1. Higher priority tiers (lower integer priority value)
    2. Higher relevance score
    3. Chronological sequence (earlier sequence preserved)
    4. Stable ID tie-breaker

    Uses a first-fit greedy approach: if a high-ranking item is too large for remaining
    budget, subsequent smaller candidate items can still be considered.
    """
    if budget <= 0:
        return [], list(blocks)

    # Sort descending by value (priority ascending, score descending, seq ascending, id ascending)
    sorted_candidates = sorted(
        blocks,
        key=lambda tb: (
            int(tb.block.priority),
            -tb.block.score,
            tb.block.seq,
            tb.block.id,
        ),
    )

    included: list[TokenizedBlock] = []
    dropped: list[TokenizedBlock] = []
    spent = 0

    for tb in sorted_candidates:
        if spent + tb.tokens <= budget:
            included.append(tb)
            spent += tb.tokens
        else:
            dropped.append(tb)

    return included, dropped

