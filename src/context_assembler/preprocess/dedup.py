"""Deduplication, state supersession, and age-based tool observation masking."""

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


def apply_observation_masking(
    blocks: Sequence[ContextBlock],
    keep_recent_turns: int | None = None,
    epoch_stride: int = 1,
) -> list[ContextBlock]:
    """Masks bulky tool_result observations older than `keep_recent_turns`.

    Inspired by SWE-agent's `LastNObservations` and OpenHands' `ObservationMaskingCondenser`,
    enhanced with `epoch_stride` snapping so the masking boundary only advances every
    `epoch_stride` turns rather than invalidating the KV prompt cache on every step.
    """
    if keep_recent_turns is None or keep_recent_turns < 0 or not blocks:
        return list(blocks)

    max_seq = max((b.seq for b in blocks), default=0)
    raw_cutoff = max_seq - keep_recent_turns
    if raw_cutoff <= 0:
        return list(blocks)

    stride = max(1, epoch_stride)
    cutoff_seq = ((raw_cutoff - 1) // stride + 1) * stride if stride > 1 else raw_cutoff

    result: list[ContextBlock] = []
    for b in blocks:
        if (
            b.kind == "tool_result"
            and not b.pinned
            and b.priority != Priority.P0
            and b.seq < cutoff_seq
            and len(b.content) > 120
        ):
            line_count = max(1, len(b.content.splitlines()))
            stub = (
                f"[Old tool observation from turn {b.seq} masked "
                f"({line_count} lines); re-run tool if needed]"
            )
            result.append(replace(b, content=stub))
        else:
            result.append(b)

    return result
