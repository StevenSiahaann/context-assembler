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
    2. Chronological conversation history turns + paired tool call/result groups
    3. Turn-local RAG / dynamic memory (sorted ascending by score so highest score is last)
    4. Active user request at the very end

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

    # Sort prefix deterministically by kind then id (system before tool_schema)
    prefix_blocks.sort(key=lambda tb: (0 if tb.block.kind == "system" else 1, tb.block.id))

    # Determine group sequence numbers so atomic tool_call / tool_result pairs stay adjacent
    max_history_seq = max((tb.block.seq for tb in history_blocks), default=0)
    group_seqs: dict[str, int] = {}
    for tb in history_blocks:
        gid = tb.block.group_id
        if gid is not None:
            group_seqs[gid] = max(group_seqs.get(gid, 0), tb.block.seq)

    def _history_sort_key(tb: TokenizedBlock) -> tuple[int, int, str, int, str]:
        b = tb.block
        eff_seq = group_seqs[b.group_id] if b.group_id is not None else b.seq
        # Recent P2 tool blocks without an explicit seq sit right after prior history turns
        is_unsequenced_recent_tool = (
            b.kind in ("tool_call", "tool_result") and eff_seq == 0 and max_history_seq > 0
        )
        tier = 1 if is_unsequenced_recent_tool else 0
        # Within the same atomic group, tool_call (0) must always precede tool_result (1)
        group_order = 0 if b.kind == "tool_call" else (1 if b.kind == "tool_result" else 0)
        return (tier, eff_seq, b.group_id or "", group_order, b.id)

    history_blocks.sort(key=_history_sort_key)

    # RAG chunks: sort by score ascending so the highest-scoring chunk sits
    # nearest to the user query at the bottom (Lost-In-The-Middle optimization).
    rag_blocks.sort(key=lambda tb: (tb.block.score, tb.block.id))

    ordered = [*prefix_blocks, *history_blocks, *rag_blocks, *active_user_blocks]

    # Calculate prefix hash over the static prefix segment to detect cache invalidations
    hasher = hashlib.blake2b(digest_size=16)
    prefix_tokens = 0
    for pb in prefix_blocks:
        hasher.update(pb.block.content.encode("utf-8"))
        prefix_tokens += pb.tokens

    return ordered, hasher.hexdigest(), prefix_tokens
