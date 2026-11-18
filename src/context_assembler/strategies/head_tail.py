"""Head-tail and middle-out truncation strategies for logs, traces, and documents."""

from __future__ import annotations

import itertools
import re
from bisect import bisect_right
from dataclasses import replace

from context_assembler.domain.block import HeadTail, TokenizedBlock
from context_assembler.security.trust_wrap import (
    format_untrusted_envelope,
    strip_untrusted_envelope,
)
from context_assembler.tokenizers.base import Tokenizer


def _chunk_text_spans(text: str, max_span_chars: int = 64) -> list[str]:
    """Splits text into word/punctuation spans, breaking giant unspaced blobs into chunks."""
    raw_parts = re.findall(r"\S+\s*|\s+", text)
    spans: list[str] = []
    for part in raw_parts:
        if len(part) <= max_span_chars:
            spans.append(part)
        else:
            for idx in range(0, len(part), max_span_chars):
                spans.append(part[idx : idx + max_span_chars])
    return spans


def _reduce_by_units(
    tblock: TokenizedBlock,
    units: list[str],
    budget: int,
    head_ratio: float,
    unit_label: str,
    sep: str,
    tokenizer: Tokenizer,
) -> TokenizedBlock | None:
    """Generic prefix-sum + binary-search reducer over lines or word/char spans."""
    total_units = len(units)
    if total_units == 0:
        return None

    unit_tokens = tokenizer.count_lines(units)
    worst_marker = f"{sep}... [{total_units} {unit_label} omitted for context budget] ...{sep}"
    overhead_probe = format_untrusted_envelope(tblock.block, worst_marker)
    overhead_cost = tokenizer.count(overhead_probe)
    usable_tokens = budget - overhead_cost

    if usable_tokens <= 0:
        return None

    head_token_target = int(usable_tokens * head_ratio)
    tail_token_target = usable_tokens - head_token_target

    head_prefix = list(itertools.accumulate(unit_tokens))
    head_idx = bisect_right(head_prefix, head_token_target)

    tail_tokens_rev = unit_tokens[::-1]
    tail_prefix = list(itertools.accumulate(tail_tokens_rev))
    tail_idx_from_end = bisect_right(tail_prefix, tail_token_target)
    tail_idx = max(head_idx, total_units - tail_idx_from_end)

    while True:
        if head_idx == 0 and tail_idx >= total_units:
            return None

        omitted_count = max(0, tail_idx - head_idx)
        marker = f"{sep}... [{omitted_count} {unit_label} omitted for context budget] ...{sep}"
        inner_joined = "".join(units[:head_idx]) + marker + "".join(units[tail_idx:])
        final_content = format_untrusted_envelope(tblock.block, inner_joined)
        final_tokens = tokenizer.count(final_content)

        if final_tokens <= budget:
            return TokenizedBlock(
                block=replace(tblock.block, content=final_content),
                tokens=final_tokens,
            )

        tail_kept = total_units - tail_idx
        if head_idx >= tail_kept and head_idx > 0:
            head_idx -= 1
        elif tail_idx < total_units:
            tail_idx += 1
        else:
            return None


def reduce_head_tail(
    tblock: TokenizedBlock,
    budget: int,
    tokenizer: Tokenizer,
) -> TokenizedBlock | None:
    """Truncates content keeping head and tail strictly within budget.

    Prefers newline boundaries first; if the input is a single massive line
    (e.g., minified JSON or unformatted dump) where line-level slicing cannot fit,
    automatically falls back to word/character span slicing.
    """
    if tblock.tokens <= budget:
        return tblock

    if budget <= 10:
        return None

    strategy = tblock.block.strategy
    head_ratio = strategy.head_ratio if isinstance(strategy, HeadTail) else 0.5

    raw_inner = strip_untrusted_envelope(tblock.block, tblock.block.content)
    lines = raw_inner.splitlines(keepends=True)
    if not lines:
        return tblock if tblock.tokens <= budget else None

    if len(lines) > 1:
        line_result = _reduce_by_units(
            tblock=tblock,
            units=lines,
            budget=budget,
            head_ratio=head_ratio,
            unit_label="lines",
            sep="\n",
            tokenizer=tokenizer,
        )
        if line_result is not None:
            return line_result

    # Fallback for single-line blobs (minified JSON, single long line)
    spans = _chunk_text_spans(raw_inner, max_span_chars=48)
    return _reduce_by_units(
        tblock=tblock,
        units=spans,
        budget=budget,
        head_ratio=head_ratio,
        unit_label="segments",
        sep=" ",
        tokenizer=tokenizer,
    )


def reduce_middle_out(
    tblock: TokenizedBlock,
    budget: int,
    tokenizer: Tokenizer,
) -> TokenizedBlock | None:
    """Preserves the beginning and end of prose or documents while eliding the middle.

    Unlike HeadTail (which prioritizes line breaks for terminal logs), MiddleOut
    operates directly on word/sentence segments so multi-sentence paragraphs
    without newlines compress smoothly.
    """
    if tblock.tokens <= budget:
        return tblock

    if budget <= 10:
        return None

    raw_inner = strip_untrusted_envelope(tblock.block, tblock.block.content)
    if not raw_inner:
        return tblock if tblock.tokens <= budget else None

    spans = _chunk_text_spans(raw_inner, max_span_chars=48)
    return _reduce_by_units(
        tblock=tblock,
        units=spans,
        budget=budget,
        head_ratio=0.5,
        unit_label="segments",
        sep=" ",
        tokenizer=tokenizer,
    )
