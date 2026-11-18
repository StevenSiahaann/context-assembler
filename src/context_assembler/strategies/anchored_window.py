"""Anchored window reduction with epoch-chunked hysteresis for conversation histories."""

from __future__ import annotations

from collections.abc import Sequence

from context_assembler.domain.block import TokenizedBlock


def _group_atomic_units(blocks: Sequence[TokenizedBlock]) -> list[list[TokenizedBlock]]:
    """Groups chronological blocks into indivisible units (single turn or atomic group_id)."""
    units: list[list[TokenizedBlock]] = []
    seen_groups: set[str] = set()

    for tb in blocks:
        gid = tb.block.group_id
        if gid is not None:
            if gid in seen_groups:
                continue
            seen_groups.add(gid)
            units.append([m for m in blocks if m.block.group_id == gid])
        else:
            units.append([tb])

    return units


def reduce_anchored_window(
    history_blocks: Sequence[TokenizedBlock],
    budget: int,
    low_watermark_ratio: float = 0.75,
) -> tuple[list[TokenizedBlock], list[TokenizedBlock]]:
    """Preserves initial goal anchor (first turn) + latest turns while dropping middle.

    Supports stateless epoch-chunked hysteresis (`low_watermark_ratio < 1.0`):
    Instead of shifting the history prefix by 1 message on every turn after budget fills,
    post-anchor history is partitioned forward from Turn 1 into fixed-size token epochs
    of `~(1 - low_watermark_ratio) * available_budget`. On overflow, whole oldest epochs
    are evicted together so the surviving history prefix stays byte-stable across
    multiple subsequent turns.
    """
    total_tokens = sum(tb.tokens for tb in history_blocks)
    if total_tokens <= budget:
        return list(history_blocks), []

    if not history_blocks or budget <= 0:
        return [], list(history_blocks)

    units = _group_atomic_units(history_blocks)
    anchor_unit = units[0]
    anchor_cost = sum(tb.tokens for tb in anchor_unit)
    if anchor_cost > budget:
        return [], list(history_blocks)

    included_set: set[str] = {tb.block.id for tb in anchor_unit}
    avail = budget - anchor_cost
    rest_units = units[1:]

    if not rest_units or avail <= 0:
        included = [tb for tb in history_blocks if tb.block.id in included_set]
        dropped = [tb for tb in history_blocks if tb.block.id not in included_set]
        return included, dropped

    # Build forward chronological epochs from Turn 1 for cache-stable chunked eviction
    if 0.1 <= low_watermark_ratio < 1.0:
        epoch_quantum = max(1, int(avail * (1.0 - low_watermark_ratio)))
        epochs: list[list[list[TokenizedBlock]]] = []
        current_epoch: list[list[TokenizedBlock]] = []
        current_cost = 0

        for u in rest_units:
            u_cost = sum(tb.tokens for tb in u)
            if current_epoch and current_cost + u_cost > epoch_quantum:
                epochs.append(current_epoch)
                current_epoch = [u]
                current_cost = u_cost
            else:
                current_epoch.append(u)
                current_cost += u_cost

        if current_epoch:
            epochs.append(current_epoch)

        # Drop whole oldest epochs from the front until remaining epochs fit in avail
        epoch_costs = [sum(tb.tokens for u in ep for tb in u) for ep in epochs]
        rem_total = sum(epoch_costs)
        start_ep = 0
        while start_ep < len(epochs) and rem_total > avail:
            rem_total -= epoch_costs[start_ep]
            start_ep += 1

        surviving_units = [u for ep in epochs[start_ep:] for u in ep]
        if surviving_units and rem_total <= avail:
            for u in surviving_units:
                for tb in u:
                    included_set.add(tb.block.id)
            included = [tb for tb in history_blocks if tb.block.id in included_set]
            dropped = [tb for tb in history_blocks if tb.block.id not in included_set]
            return included, dropped

    # Fallback (or when low_watermark_ratio == 1.0): greedy newest-first unit packing
    spent = 0
    for u in reversed(rest_units):
        u_cost = sum(tb.tokens for tb in u)
        if spent + u_cost <= avail:
            for tb in u:
                included_set.add(tb.block.id)
            spent += u_cost

    included = [tb for tb in history_blocks if tb.block.id in included_set]
    dropped = [tb for tb in history_blocks if tb.block.id not in included_set]

    return included, dropped
