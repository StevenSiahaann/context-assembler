"""Head-tail and middle-out truncation strategies for logs, traces, and documents."""

from __future__ import annotations

import itertools
from bisect import bisect_right
from dataclasses import replace

from context_assembler.domain.block import HeadTail, TokenizedBlock
from context_assembler.security.trust_wrap import (
    format_untrusted_envelope,
    strip_untrusted_envelope,
)
from context_assembler.tokenizers.base import Tokenizer


def reduce_head_tail(
    tblock: TokenizedBlock,
    budget: int,
    tokenizer: Tokenizer,
) -> TokenizedBlock | None:
    """Truncates content keeping head lines and tail lines strictly within budget.

    Splits at newline boundaries and uses prefix sums + binary search
    to determine how many lines fit without repeatedly re-tokenizing, followed by
    an exact token verification step to guard against BPE boundary merges.
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

    total_lines = len(lines)
    line_tokens = tokenizer.count_lines(lines)

    # Measure exact worst-case overhead of the omission marker + any trust envelope
    worst_marker = f"\n... [{total_lines} lines omitted for context budget] ...\n"
    overhead_probe = format_untrusted_envelope(tblock.block, worst_marker)
    overhead_cost = tokenizer.count(overhead_probe)
    usable_tokens = budget - overhead_cost

    if usable_tokens <= 0:
        return None

    head_token_target = int(usable_tokens * head_ratio)
    tail_token_target = usable_tokens - head_token_target

    head_prefix = list(itertools.accumulate(line_tokens))
    head_idx = bisect_right(head_prefix, head_token_target)

    tail_tokens_rev = line_tokens[::-1]
    tail_prefix = list(itertools.accumulate(tail_tokens_rev))
    tail_idx_from_end = bisect_right(tail_prefix, tail_token_target)
    tail_idx = max(head_idx, total_lines - tail_idx_from_end)

    # Guard against BPE merge drift across concatenation boundaries
    while True:
        if head_idx == 0 and tail_idx >= total_lines:
            return None

        omitted_count = max(0, tail_idx - head_idx)
        marker = f"\n... [{omitted_count} lines omitted for context budget] ...\n"
        inner_joined = "".join(lines[:head_idx]) + marker + "".join(lines[tail_idx:])
        final_content = format_untrusted_envelope(tblock.block, inner_joined)
        final_tokens = tokenizer.count(final_content)

        if final_tokens <= budget:
            return TokenizedBlock(
                block=replace(tblock.block, content=final_content),
                tokens=final_tokens,
            )

        # Step back one line from whichever side currently holds more lines
        tail_kept = total_lines - tail_idx
        if head_idx >= tail_kept and head_idx > 0:
            head_idx -= 1
        elif tail_idx < total_lines:
            tail_idx += 1
        else:
            return None
