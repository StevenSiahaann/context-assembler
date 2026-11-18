"""Budget policies, slot definitions, and model runtime profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from context_assembler.domain.block import BlockKind


@dataclass(frozen=True, slots=True)
class SlotPolicy:
    """Resource allocation policy for a specific category of blocks.

    Attributes:
        kind: Target block category (e.g. 'rag', 'tool_result', 'history').
        min_tokens: Guaranteed floor capacity if content exists.
        max_tokens: Hard ceiling cap that content in this slot may not exceed.
        weight: Relative weight for distributing surplus budget in water-filling.
    """

    kind: BlockKind
    min_tokens: int
    max_tokens: int
    weight: int = 1
    low_watermark_ratio: float = 0.75

    def __post_init__(self) -> None:
        if self.min_tokens < 0:
            raise ValueError(f"min_tokens must be non-negative, got {self.min_tokens}")
        if self.max_tokens < self.min_tokens:
            raise ValueError(
                f"max_tokens ({self.max_tokens}) cannot be smaller "
                f"than min_tokens ({self.min_tokens})"
            )
        if self.weight <= 0:
            raise ValueError(f"weight must be strictly positive, got {self.weight}")
        if not 0.1 <= self.low_watermark_ratio <= 1.0:
            raise ValueError(
                f"low_watermark_ratio must be in [0.1, 1.0], got {self.low_watermark_ratio}"
            )


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """Operational parameters of the target language model.

    Attributes:
        context_window: Total token limit of the model architecture.
        max_output_tokens: Reserved space for the generation response.
        per_message_overhead: Protocol framing tokens consumed per message envelope.
        safety_margin_ratio: Buffer percentage to absorb local tokenizer drift.
        reasoning_budget_tokens: Reserved capacity for chain-of-thought/thinking tokens.
        system_role: Role name used for top-level instructions ('system' or 'developer').
        requires_alternation: Whether the provider strictly mandates user/assistant turns.
        max_cache_breakpoints: Explicit prompt cache markers allowed (e.g. 4 for Anthropic).
        tool_schema_overhead: Estimated framing overhead per tool definition.
    """

    context_window: int
    max_output_tokens: int
    per_message_overhead: int = 4
    safety_margin_ratio: float = 0.03
    reasoning_budget_tokens: int = 0
    system_role: Literal["system", "developer"] = "system"
    requires_alternation: bool = False
    max_cache_breakpoints: int = 0
    tool_schema_overhead: int = 12

    def __post_init__(self) -> None:
        if self.context_window <= 0:
            raise ValueError(f"context_window must be positive, got {self.context_window}")
        if self.max_output_tokens <= 0:
            raise ValueError(f"max_output_tokens must be positive, got {self.max_output_tokens}")
        if not 0.0 <= self.safety_margin_ratio < 0.5:
            raise ValueError(
                f"safety_margin_ratio must be between 0.0 and 0.5, got {self.safety_margin_ratio}"
            )
        total_reserved = self.max_output_tokens + self.reasoning_budget_tokens
        if total_reserved >= self.context_window:
            raise ValueError(
                f"Reserved output tokens ({total_reserved}) cannot exceed "
                f"context window ({self.context_window})"
            )
