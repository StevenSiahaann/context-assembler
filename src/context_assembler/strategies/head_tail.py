"""Head-tail truncation strategy for structured logs, traces, and tool results."""

from __future__ import annotations

import itertools
from bisect import bisect_right
from dataclasses import replace

from context_assembler.domain.block import HeadTail, TokenizedBlock
from context_assembler.tokenizers.base import Tokenizer


def reduce_head_tail(
    tblock: TokenizedBlock,
    budget: int,
    tokenizer: Tokenizer,
) -> TokenizedBlock | None:
    """Truncates content keeping head lines and tail lines within budget.

    Splits at newline boundaries and uses prefix sums + binary search
    to determine exactly how many lines fit without repeatedly re-tokenizing.
    """
    if tblock.tokens <= budget:
        return tblock

    if budget <= 10:
        # Budget too tiny to even display an informative omission message
        return None

    strategy = tblock.block.strategy
    head_ratio = strategy.head_ratio if isinstance(strategy, HeadTail) else 0.5

    lines = tblock.block.content.splitlines(keepends=True)
    if not lines:
        return tblock

    # Calculate tokens per line
    line_tokens = tokenizer.count_lines(lines)
    total_lines = len(lines)

    # Estimate marker cost first so it's guaranteed within budget
    sample_marker = "\n... [9999 lines omitted] ...\n"
    marker_cost = tokenizer.count(sample_marker)
    usable_tokens = budget - marker_cost

    if usable_tokens <= 0:
        return None

    head_token_target = int(usable_tokens * head_ratio)
    tail_token_target = usable_tokens - head_token_target

    # Prefix sums from the head
    head_prefix = list(itertools.accumulate(line_tokens))
    head_idx = bisect_right(head_prefix, head_token_target)

    # Suffix sums from the tail (reverse prefix sums)
    tail_tokens_rev = line_tokens[::-1]
    tail_prefix = list(itertools.accumulate(tail_tokens_rev))
    tail_idx_from_end = bisect_right(tail_prefix, tail_token_target)
    tail_idx = max(head_idx, total_lines - tail_idx_from_end)

    omitted_count = max(0, tail_idx - head_idx)
    marker = f"\n... [{omitted_count} lines omitted for context budget] ...\n"

    new_content = "".join(lines[:head_idx]) + marker + "".join(lines[tail_idx:])
    final_tokens = tokenizer.count(new_content)

    return TokenizedBlock(
        block=replace(tblock.block, content=new_content),
        tokens=final_tokens,
    )
