"""Deduplication and state supersession preprocessing."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import replace

from context_assembler.domain.block import ContextBlock, Priority


def deduplicate_exact(blocks: Sequence[ContextBlock]) -> list[ContextBlock]:
    """Removes identical duplicate RAG/memory blocks, retaining the highest score.

    Conversation history, system instructions, user queries, and grouped tool blocks
    are never deduplicated by content hash since repeated phrases (e.g. "OK") are
    valid chronological turns.
    """
    seen: dict[bytes, ContextBlock] = {}
    ordered_hashes: list[bytes] = []

    for i, b in enumerate(blocks):
        is_dedup_candidate = (
            b.kind in ("rag", "memory")
            and b.group_id is None
            and not b.pinned
            and b.priority != Priority.P0
        )
        if not is_dedup_candidate:
            h = f"unique:{i}:{b.id}".encode()
            seen[h] = b
            ordered_hashes.append(h)
            continue

        h = hashlib.blake2b(b.content.encode("utf-8"), digest_size=16).digest()
        if h in seen:
            # Keep whichever version has higher relevance score (or higher priority)
            prev = seen[h]
            if (int(b.priority), -b.score) < (int(prev.priority), -prev.score):
                seen[h] = b
        else:
            seen[h] = b
            ordered_hashes.append(h)

    return [seen[h] for h in ordered_hashes]


def apply_supersession(blocks: Sequence[ContextBlock]) -> list[ContextBlock]:
    """Replaces older versions of superseded state blocks with a lightweight stub.

    Example: Reading `package.json` three times across 10 turns. Only the newest
    read carries the payload; earlier turns retain a stub so pairing remains intact.
    """
    latest_seq_per_key: dict[str, int] = {}

    for b in blocks:
        if b.supersedes_key:
            latest_seq_per_key[b.supersedes_key] = max(
                latest_seq_per_key.get(b.supersedes_key, -1),
                b.seq,
            )

    result: list[ContextBlock] = []
    for b in blocks:
        if b.supersedes_key:
            latest = latest_seq_per_key[b.supersedes_key]
            if b.seq < latest:
                stub = f"[Content superseded by later read of key: {b.supersedes_key}]"
                result.append(replace(b, content=stub))
                continue
        result.append(b)

    return result
