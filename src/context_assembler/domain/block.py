"""Core context block definitions and reduction strategy representations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Literal

from context_assembler.domain.errors import InvalidBlockError


class Priority(IntEnum):
    """Priority tiers determining eviction order under budget pressure.

    Lower integer value means higher priority (P0 is most critical, P4 is first
    to be considered for eviction or truncation).
    """

    P0 = 0  # Non-negotiable: system prompt, tool definitions
    P1 = 1  # Active user request / intent
    P2 = 2  # Recent tool execution results
    P3 = 3  # RAG retrieved documents & core working memory
    P4 = 4  # Historical conversation turns


BlockKind = Literal[
    "system",
    "tool_schema",
    "user",
    "memory",
    "rag",
    "tool_call",
    "tool_result",
    "history",
]

Role = Literal["system", "user", "assistant", "tool", "developer"]


@dataclass(frozen=True, slots=True)
class Never:
    """Do not reduce or drop. If budget fails, raise P0OverflowError."""


@dataclass(frozen=True, slots=True)
class Drop:
    """Drop entire block when budget is insufficient."""


@dataclass(frozen=True, slots=True)
class HeadTail:
    """Retain starting and ending lines with an omission marker in between.

    Useful for logs, stack traces, and large query dumps where both the initial
    command context and final error status matter most.
    """

    head_ratio: float = 0.5


@dataclass(frozen=True, slots=True)
class MiddleOut:
    """Preserve head and tail while trimming the center (lost-in-the-middle)."""


@dataclass(frozen=True, slots=True)
class Alternatives:
    """Ordered fallback representations (e.g. full doc -> executive summary)."""

    variants: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Summarize:
    """Flag block as needing external summarization when under budget strain."""

    hook_id: str


ReductionStrategy = Never | Drop | HeadTail | MiddleOut | Alternatives | Summarize


@dataclass(frozen=True, slots=True)
class ContextBlock:
    """Atomic unit of prompt material submitted to the assembler.

    Attributes:
        id: Stable identifier, also acts as a deterministic tie-breaker.
        kind: Semantic source category for slot policy routing.
        role: Target message role for chat completion APIs.
        content: Raw textual payload.
        priority: Eviction priority tier.
        strategy: How to compress this block if it exceeds available space.
        score: Relevance or importance metric (higher is more relevant).
        group_id: ID linking atomic pairs (e.g. tool call and its response).
        cacheable: Whether this block belongs in the static cached prefix.
        seq: Logical chronological sequence index for conversation turns.
        trust: Trust boundary label. Untrusted content is forbidden in system roles.
        source: Provenance locator (e.g. file path, URL, tool name).
        supersedes_key: State key where only the newest sequence version is kept.
        pinned: If true, block cannot be dropped unless P0 overflows.
        fixed_tokens: Explicit token count override (useful for pre-computed tokens).
    """

    id: str
    kind: BlockKind
    role: Role
    content: str
    priority: Priority
    strategy: ReductionStrategy
    score: float = 0.0
    group_id: str | None = None
    cacheable: bool = False
    seq: int = 0
    trust: Literal["trusted", "untrusted"] = "trusted"
    source: str | None = None
    supersedes_key: str | None = None
    pinned: bool = False
    fixed_tokens: int | None = None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise InvalidBlockError(self.id, "Block ID cannot be empty.")

        if (
            isinstance(self.strategy, HeadTail)
            and not (0.0 < self.strategy.head_ratio < 1.0)
        ):
            raise InvalidBlockError(
                self.id,
                f"head_ratio must be strictly between 0 and 1, got {self.strategy.head_ratio}",
            )

        # Enforce OWASP LLM01: external untrusted content must never inhabit system prompt
        if self.trust == "untrusted" and self.role in ("system", "developer"):
            raise InvalidBlockError(
                self.id,
                "Untrusted input (RAG/tool/web) cannot be assigned to system or developer roles.",
            )


@dataclass(frozen=True, slots=True)
class TokenizedBlock:
    """ContextBlock paired with its pre-calculated token measurement."""

    block: ContextBlock
    tokens: int
