"""Tokenizer interfaces and abstract protocols."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    """Protocol for token counting and conversion implementations."""

    @property
    def name(self) -> str:
        """Unique identifier representing the underlying tokenizer encoding."""
        ...

    def count(self, text: str) -> int:
        """Return the exact or estimated token count for the given text."""
        ...

    def count_lines(self, lines: list[str]) -> list[int]:
        """Return token count for each line in batch.

        Default implementation counts line-by-line, but custom adapters can optimize.
        """
        ...

