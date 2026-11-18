"""Structural sanitization and boundary wrapping for untrusted external inputs."""

from __future__ import annotations

from dataclasses import replace

from context_assembler.domain.block import TokenizedBlock
from context_assembler.tokenizers.base import Tokenizer


def wrap_untrusted_block(
    tb: TokenizedBlock,
    tokenizer: Tokenizer | None = None,
) -> TokenizedBlock:
    """Encloses untrusted context (RAG, web results, tool output) in protective delimiters.

    Mitigates prompt injection (OWASP LLM01) by clearly distinguishing data payload
    from model instruction channels, escaping any embedded closing delimiter tags
    to block syntax breakout attacks.
    """
    if tb.block.trust != "untrusted":
        return tb

    # Neutralize breakout attempts by escaping closing delimiter tags
    sanitized_content = tb.block.content.replace("</document>", "&lt;/document&gt;")

    source_attr = f' source="{tb.block.source}"' if tb.block.source else ""
    wrapped = (
        f'<document id="{tb.block.id}"{source_attr} trust="untrusted">\n'
        f"{sanitized_content}\n"
        f"</document>"
    )

    new_tokens = tokenizer.count(wrapped) if tokenizer else tb.tokens + 10

    return TokenizedBlock(
        block=replace(tb.block, content=wrapped),
        tokens=new_tokens,
    )
