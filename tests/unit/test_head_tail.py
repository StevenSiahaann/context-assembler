"""Unit tests for head-tail truncation."""

from context_assembler.domain.block import (
    ContextBlock,
    HeadTail,
    Priority,
    TokenizedBlock,
)
from context_assembler.strategies.head_tail import reduce_head_tail
from context_assembler.tokenizers.heuristic import HeuristicTokenizer


def test_head_tail_truncates_long_logs_within_budget() -> None:
    tokenizer = HeuristicTokenizer(chars_per_token=4.0)

    # 100 lines of log output
    lines = [f"2026-11-18 22:00:{i:02d} INFO Process event payload status=OK\n" for i in range(100)]
    full_text = "".join(lines)
    full_tokens = tokenizer.count(full_text)

    block = ContextBlock(
        id="trace_log_1",
        kind="tool_result",
        role="tool",
        content=full_text,
        priority=Priority.P2,
        strategy=HeadTail(head_ratio=0.5),
    )
    tblock = TokenizedBlock(block, full_tokens)

    # Truncate to tight budget of 150 tokens
    budget = 150
    result = reduce_head_tail(tblock, budget, tokenizer)

    assert result is not None
    assert result.tokens <= budget
    assert "lines omitted for context budget" in result.block.content
    # Confirm starting and ending markers are intact
    assert lines[0] in result.block.content
    assert lines[-1] in result.block.content


def test_head_tail_returns_none_if_budget_too_small() -> None:
    tokenizer = HeuristicTokenizer()
    block = ContextBlock(
        id="b1",
        kind="tool_result",
        role="tool",
        content="some log line\nanother line\n",
        priority=Priority.P2,
        strategy=HeadTail(0.5),
    )
    tblock = TokenizedBlock(block, 20)
    # Budget of only 5 tokens is smaller than the omission marker itself
    result = reduce_head_tail(tblock, budget=5, tokenizer=tokenizer)
    assert result is None

