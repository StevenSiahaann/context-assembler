"""Lightweight character/word heuristic tokenizer for offline environments and testing."""

from __future__ import annotations

import math


class HeuristicTokenizer:
    """Zero-dependency heuristic tokenizer based on character ratios.

    Defaults to ~3.8 characters per token for English text and code,
    providing a conservative upper-bound for testing and fallback use cases.
    """

    def __init__(self, chars_per_token: float = 3.8, name: str = "heuristic") -> None:
        if chars_per_token <= 0:
            raise ValueError(f"chars_per_token must be positive, got {chars_per_token}")
        self._chars_per_token = chars_per_token
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def count(self, text: str) -> int:
        if not text:
            return 0
        return max(1, math.ceil(len(text) / self._chars_per_token))

    def count_lines(self, lines: list[str]) -> list[int]:
        return [self.count(line) for line in lines]
