"""Domain exceptions for context assembly.

We use a typed hierarchy rather than generic ValueErrors so callers
can cleanly catch specific recovery conditions (e.g. prompt trimming
vs unrecoverable P0 overflow).
"""

from __future__ import annotations


class AssemblyError(Exception):
    """Base error for all context-assembler failures."""


class P0OverflowError(AssemblyError):
    """Raised when non-negotiable P0 blocks exceed the entire input budget.

    System prompts and tool schemas cannot be dropped without breaking
    agent behavior, so this must bubble up to caller to trim prompt definitions.
    """

    def __init__(self, required: int, available: int) -> None:
        super().__init__(
            f"P0 blocks require {required} tokens, but only {available} tokens "
            f"are available in the input budget."
        )
        self.required = required
        self.available = available


class InvalidBlockError(AssemblyError):
    """Raised when a block violates domain invariants (e.g. empty id, bad role)."""

    def __init__(self, block_id: str, reason: str) -> None:
        super().__init__(f"Invalid block {block_id!r}: {reason}")
        self.block_id = block_id
        self.reason = reason


class OrphanToolResultError(AssemblyError):
    """Raised when a tool_result arrives without its corresponding tool_call."""

    def __init__(self, group_id: str) -> None:
        super().__init__(
            f"Tool result in group {group_id!r} has no corresponding tool call. "
            f"Providers will reject unpaired tool messages."
        )
        self.group_id = group_id


class StructuralValidityError(AssemblyError):
    """Raised when message sequencing breaks provider protocol invariants."""

    def __init__(self, rule: str, detail: str) -> None:
        super().__init__(f"Structural validity rule {rule!r} violated: {detail}")
        self.rule = rule
        self.detail = detail

