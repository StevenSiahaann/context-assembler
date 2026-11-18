"""Deduplication and state supersession preprocessing."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import replace

from context_assembler.domain.block import ContextBlock


def deduplicate_exact(blocks: Sequence[ContextBlock]) -> list[ContextBlock]:
    """Removes identical duplicate blocks, retaining the one with highest score."""
    seen: dict[bytes, ContextBlock] = {}
    ordered_hashes: list[bytes] = []

    for b in blocks:
        # P0 blocks and user prompts are never deduplicated
        if b.kind in ("system", "tool_schema", "user") or b.group_id is not None:
            h = f"unique:{b.id}".encode()
            seen[h] = b
            ordered_hashes.append(h)
            continue

        h = hashlib.blake2b(b.content.encode("utf-8"), digest_size=16).digest()
        if h in seen:
            # Keep whichever version has higher relevance score
            if b.score > seen[h].score:
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
                # Replace with placeholder stub
                stub = f"[Content superseded by later read of key: {b.supersedes_key}]"
                result.append(replace(b, content=stub))
                continue
        result.append(b)

    return result
