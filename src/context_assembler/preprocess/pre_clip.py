"""Pre-tokenization character clipping for massive inputs."""

from __future__ import annotations

from dataclasses import replace

from context_assembler.domain.block import ContextBlock, HeadTail


def pre_clip_block(block: ContextBlock, max_chars: int = 150_000) -> ContextBlock:
    """Clips excessively large blocks before tokenization.

    Tokenizing 50MB of raw terminal output or minified JS produces severe latency spikes.
    If a block exceeds max_chars and has a HeadTail strategy, we coarsely slice it
    first with a safety multiplier, letting subsequent token-level truncation do the fine trim.
    """
    if len(block.content) <= max_chars or not isinstance(block.strategy, HeadTail):
        return block

    ratio = block.strategy.head_ratio
    head_len = int(max_chars * ratio)
    tail_len = max_chars - head_len

    head = block.content[:head_len]
    tail = block.content[-tail_len:]
    diff = len(block.content) - max_chars
    marker = f"\n\n... [coarse pre-clip: {diff} characters omitted] ...\n\n"

    return replace(block, content=head + marker + tail)
