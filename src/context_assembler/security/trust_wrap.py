"""Structural sanitization and boundary wrapping for untrusted external inputs."""

from __future__ import annotations

from dataclasses import replace

from context_assembler.domain.block import ContextBlock, TokenizedBlock
from context_assembler.tokenizers.base import Tokenizer

_OPEN_PREFIX = '<document id="'
_CLOSE_TAG = "</document>"


def format_untrusted_envelope(block: ContextBlock, inner_content: str) -> str:
    """Formats untrusted content inside a <document> provenance envelope.

    Idempotent if inner_content is already wrapped for this block ID.
    Escapes any embedded closing tags to prevent delimiter breakout (OWASP LLM01).
    """
    if block.trust != "untrusted":
        return inner_content

    expected_open = f'{_OPEN_PREFIX}{block.id}"'
    if inner_content.startswith(expected_open) and inner_content.endswith(_CLOSE_TAG):
        return inner_content

    sanitized = inner_content.replace(_CLOSE_TAG, "&lt;/document&gt;")
    source_attr = f' source="{block.source}"' if block.source else ""
    return (
        f'<document id="{block.id}"{source_attr} trust="untrusted">\n'
        f"{sanitized}\n"
        f"{_CLOSE_TAG}"
    )


def strip_untrusted_envelope(block: ContextBlock, content: str) -> str:
    """Extracts the raw payload from an already-wrapped untrusted block if present."""
    if block.trust != "untrusted":
        return content
    expected_open = f'{_OPEN_PREFIX}{block.id}"'
    if content.startswith(expected_open) and content.endswith(_CLOSE_TAG):
        first_newline = content.find("\n")
        if first_newline != -1:
            return content[first_newline + 1 : -len(_CLOSE_TAG) - 1]
    return content


def wrap_untrusted_block(
    tb: TokenizedBlock,
    tokenizer: Tokenizer | None = None,
) -> TokenizedBlock:
    """Encloses untrusted context (RAG, web results, tool output) in protective delimiters."""
    if tb.block.trust != "untrusted":
        return tb

    wrapped = format_untrusted_envelope(tb.block, tb.block.content)
    if wrapped == tb.block.content:
        return tb

    new_tokens = tokenizer.count(wrapped) if tokenizer else tb.tokens + 10
    return TokenizedBlock(
        block=replace(tb.block, content=wrapped),
        tokens=new_tokens,
    )
