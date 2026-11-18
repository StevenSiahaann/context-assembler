"""Water-filling priority budget allocator.

Fairly partitions dynamic token budget across competing context slots
(RAG, history, memory, tool outputs) honoring minimum floors, maximum caps,
and relative weight multipliers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from context_assembler.domain.budget import SlotPolicy


def water_fill(
    budget: int,
    demand: Mapping[str, int],
    policy: Mapping[str, SlotPolicy],
    order: Sequence[str],
) -> dict[str, int]:
    """Allocates available budget across requested slots.

    Args:
        budget: Total tokens available to distribute across dynamic slots.
        demand: Actual token demand per slot kind.
        policy: Configured SlotPolicy per slot kind.
        order: Deterministic sequence of slot kinds (highest priority first).

    Returns:
        Mapping of slot kind to allocated token budget.
    """
    if budget <= 0:
        return {k: 0 for k in order}

    # Effective ceiling for each slot is min(policy.max, actual demand)
    cap = {k: min(policy[k].max_tokens, max(0, demand.get(k, 0))) for k in order}
    alloc: dict[str, int] = {k: 0 for k in order}
    remaining = budget

    # Phase 1: Guarantee minimum floor allocations in priority order
    for k in order:
        if remaining <= 0:
            break
        floor = min(policy[k].min_tokens, cap[k], remaining)
        alloc[k] = floor
        remaining -= floor

    # Phase 2: Proportional water-filling for remaining headroom
    active = [k for k in order if alloc[k] < cap[k]]

    while remaining > 0 and active:
        total_w = sum(policy[k].weight for k in active)
        if total_w <= 0:
            break

        given = 0
        for k in active:
            needed = cap[k] - alloc[k]
            share = min(needed, (remaining * policy[k].weight) // total_w)
            alloc[k] += share
            given += share

        if given == 0:
            # When integer division rounds down to zero, distribute remaining tokens
            # one-by-one to slots in strict priority order.
            for k in active:
                if remaining <= 0:
                    break
                if alloc[k] < cap[k]:
                    alloc[k] += 1
                    remaining -= 1
            break

        remaining -= given
        active = [k for k in order if alloc[k] < cap[k]]

    return alloc
