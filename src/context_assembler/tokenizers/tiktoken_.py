"""Tiktoken tokenizer adapter with robust special token handling."""

from __future__ import annotations

from typing import Any

try:
    import tiktoken  # pyright: ignore[reportMissingImports]
except ImportError:  # pragma: no cover
    tiktoken = None


class TiktokenTokenizer:
    """Tokenizer adapter backed by OpenAI's official tiktoken library."""

    def __init__(self, encoding_name: str = "o200k_base") -> None:
        if tiktoken is None:
            raise ImportError(
                "tiktoken is required for TiktokenTokenizer. Install via "
                "`pip install 'context-assembler[openai]'` or `pip install tiktoken`."
            )
        self._encoding_name = encoding_name
        # tiktoken returns an Encoding instance lacking full pyright stub annotations
        self._encoding: Any = tiktoken.get_encoding(encoding_name)  # pyright: ignore[reportUnknownMemberType]

    @property
    def name(self) -> str:
        return f"tiktoken:{self._encoding_name}"

    def count(self, text: str) -> int:
        if not text:
            return 0
        # By setting disallowed_special=(), we treat control markers like <|endoftext|>
        # as plain text rather than raising DisallowedSpecial exceptions on user input.
        tokens: list[int] = self._encoding.encode(text, disallowed_special=())
        return len(tokens)

    def count_lines(self, lines: list[str]) -> list[int]:
        if not lines:
            return []
        res: list[int] = []
        for line in lines:
            t: list[int] = self._encoding.encode(line, disallowed_special=())
            res.append(len(t))
        return res
