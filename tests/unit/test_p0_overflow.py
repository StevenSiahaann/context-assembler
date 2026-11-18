"""Unit tests for P0 overflow error reporting."""

import pytest

from context_assembler.assembler import Assembler
from context_assembler.domain.block import (
    ContextBlock,
    Never,
    Priority,
)
from context_assembler.domain.budget import ModelProfile
from context_assembler.domain.errors import P0OverflowError
from context_assembler.tokenizers.heuristic import HeuristicTokenizer


def test_p0_overflow_raises_typed_exception_with_exact_deficit() -> None:
    # 500 token model, 200 reserved output -> 300 gross -> ~280 input budget
    model = ModelProfile(
        context_window=500,
        max_output_tokens=200,
        per_message_overhead=4,
        safety_margin_ratio=0.05,
    )
    tokenizer = HeuristicTokenizer(chars_per_token=1.0)
    assembler = Assembler(model=model, tokenizer=tokenizer)

    # 400 token system prompt (exceeds 280 input budget)
    giant_system_prompt = ContextBlock(
        id="sys_giant",
        kind="system",
        role="system",
        content="x" * 400,
        priority=Priority.P0,
        strategy=Never(),
    )

    with pytest.raises(P0OverflowError) as exc_info:
        assembler.assemble([giant_system_prompt])

    err = exc_info.value
    assert err.required == 400
    assert err.available < 400
    assert "P0 blocks require 400 tokens" in str(err)

