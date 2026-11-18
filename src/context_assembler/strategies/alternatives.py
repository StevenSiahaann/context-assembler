"""Alternatives fallback strategy (inspired by priompt <first>)."""

from __future__ import annotations

from dataclasses import replace

from context_assembler.domain.block import Alternatives, TokenizedBlock
from context_assembler.security.trust_wrap import format_untrusted_envelope
from context_assembler.tokenizers.base import Tokenizer


def reduce_alternatives(
    tblock: TokenizedBlock,
    budget: int,
    tokenizer: Tokenizer,
) -> TokenizedBlock | None:
    """Selects the first representation variant that fits within budget."""
    if tblock.tokens <= budget:
        return tblock

    strategy = tblock.block.strategy
    if not isinstance(strategy, Alternatives):
        return None

    for variant in strategy.variants:
        wrapped = format_untrusted_envelope(tblock.block, variant)
        cost = tokenizer.count(wrapped)
        if cost <= budget:
            return TokenizedBlock(
                block=replace(tblock.block, content=wrapped),
                tokens=cost,
            )

    return None
