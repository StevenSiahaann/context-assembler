"""Structured reporting and observability telemetry for context assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ActionType = Literal["included", "truncated", "dropped", "needs_summary"]

DecisionReason = Literal[
    "fits",
    "slot_budget_exceeded",
    "lower_score",
    "group_dropped",
    "outside_window",
    "duplicate",
    "superseded",
    "structural_fix",
    "pre_clipped",
]


@dataclass(frozen=True, slots=True)
class BlockDecision:
    """Audit entry recording the treatment of an individual block."""

    block_id: str
    action: ActionType
    tokens_before: int
    tokens_after: int
    reason: DecisionReason


@dataclass(frozen=True, slots=True)
class SlotUsage:
    """Capacity utilization metrics for a given slot category."""

    allocated: int
    used: int
    demand: int


@dataclass(frozen=True, slots=True)
class AssemblyReport:
    """Comprehensive diagnostic breakdown of the completed assembly pass.

    Enables telemetry dashboards, regression detection, and automated alerting
    on prompt drift and context exhaustion.
    """

    input_budget: int
    total_tokens: int
    utilization: float
    per_slot: dict[str, SlotUsage]
    cache_prefix_tokens: int
    prefix_hash: str
    tool_schema_tokens: dict[str, int]
    warnings: tuple[str, ...]
    decisions: tuple[BlockDecision, ...]
    duration_ms: float

