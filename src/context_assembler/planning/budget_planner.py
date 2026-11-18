"""Input budget planning based on model parameters and safety buffers."""

from __future__ import annotations

import math
from dataclasses import dataclass

from context_assembler.domain.budget import ModelProfile


@dataclass(frozen=True, slots=True)
class PlannedBudget:
    """Detailed breakdown of allocated and reserved token capacities."""

    context_window: int
    reserved_output: int
    reserved_reasoning: int
    safety_margin_tokens: int
    framing_overhead: int
    input_budget: int


def calculate_input_budget(
    model: ModelProfile,
    estimated_messages: int = 0,
    tool_count: int = 0,
) -> PlannedBudget:
    """Calculates safe input budget available for prompt assembly.

    Deducts output generation reserves, chain-of-thought capacity, framing overhead,
    and a percentage-based safety margin to protect against tokenizer drift.
    """
    reserved_generation = model.max_output_tokens + model.reasoning_budget_tokens
    available_gross = model.context_window - reserved_generation

    if available_gross <= 0:
        return PlannedBudget(
            context_window=model.context_window,
            reserved_output=model.max_output_tokens,
            reserved_reasoning=model.reasoning_budget_tokens,
            safety_margin_tokens=0,
            framing_overhead=0,
            input_budget=0,
        )

    safety_margin = math.ceil(available_gross * model.safety_margin_ratio)
    framing_overhead = (
        max(0, estimated_messages) * model.per_message_overhead
        + max(0, tool_count) * model.tool_schema_overhead
    )

    input_budget = max(0, available_gross - safety_margin - framing_overhead)

    return PlannedBudget(
        context_window=model.context_window,
        reserved_output=model.max_output_tokens,
        reserved_reasoning=model.reasoning_budget_tokens,
        safety_margin_tokens=safety_margin,
        framing_overhead=framing_overhead,
        input_budget=input_budget,
    )

