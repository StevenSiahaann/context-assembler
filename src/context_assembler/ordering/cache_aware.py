"""Cache-aware block ordering and prefix breakpoint assignment."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from context_assembler.domain.block import TokenizedBlock


def order_blocks_for_caching(
    blocks: Sequence[TokenizedBlock],
) -> tuple[list[TokenizedBlock], str, int]:
    """Arranges blocks to maximize KV prompt-cache hits across turns.

    Layout hierarchy:
    1. Static prefix: system instructions, tool definitions, pinned foundational memory
    2. Chronological conversation history turns
    3. Latest active user turn, placing contextual RAG documents immediately
       before the final user question to optimize attention and reasoning.

    Returns:
        tuple of (ordered_blocks, prefix_hash, cacheable_prefix_tokens)
    """
    prefix_blocks: list[TokenizedBlock] = []
    history_blocks: list[TokenizedBlock] = []
    rag_blocks: list[TokenizedBlock] = []
    active_user_blocks: list[TokenizedBlock] = []

    for tb in blocks:
        is_prefix = (
            tb.block.role in ("system", "developer")
            or tb.block.kind == "tool_schema"
            or (tb.block.pinned and tb.block.cacheable)
        )
        if is_prefix:
            prefix_blocks.append(tb)
        elif tb.block.kind in ("rag", "memory") and tb.block.role == "user":
            rag_blocks.append(tb)
        elif tb.block.priority == 1 and tb.block.role == "user":
            active_user_blocks.append(tb)
        else:
            history_blocks.append(tb)

    # Sort prefix deterministically by id
    prefix_blocks.sort(key=lambda tb: tb.block.id)

    # History follows chronological order
    history_blocks.sort(key=lambda tb: (tb.block.seq, tb.block.id))

    # RAG chunks: sort by score ascending so the highest-scoring chunk sits
    # nearest to the user query at the bottom (Lost-In-The-Middle optimization).
    rag_blocks.sort(key=lambda tb: tb.block.score)

    ordered = [*prefix_blocks, *history_blocks, *rag_blocks, *active_user_blocks]

    # Calculate prefix hash over the static prefix segment to detect cache invalidations
    hasher = hashlib.blake2b(digest_size=16)
    prefix_tokens = 0
    for pb in prefix_blocks:
        hasher.update(pb.block.content.encode("utf-8"))
        prefix_tokens += pb.tokens

    return ordered, hasher.hexdigest(), prefix_tokens
