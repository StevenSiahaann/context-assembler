"""Performance benchmarks for context assembly."""

import pytest

from context_assembler.assembler import Assembler
from context_assembler.domain.block import (
    ContextBlock,
    Drop,
    HeadTail,
    Never,
    Priority,
)
from context_assembler.domain.budget import ModelProfile
from context_assembler.tokenizers.cached import CachedTokenizer
from context_assembler.tokenizers.heuristic import HeuristicTokenizer


@pytest.fixture
def sample_blocks_200() -> list[ContextBlock]:
    blocks: list[ContextBlock] = [
        ContextBlock(
            id="sys_main",
            kind="system",
            role="system",
            content="You are an enterprise assistant.",
            priority=Priority.P0,
            strategy=Never(),
            cacheable=True,
        ),
        ContextBlock(
            id="user_active",
            kind="user",
            role="user",
            content="Generate the database migration plan for users and orders.",
            priority=Priority.P1,
            strategy=HeadTail(0.5),
        ),
    ]

    # Add 100 RAG documents
    for i in range(100):
        blocks.append(
            ContextBlock(
                id=f"rag_doc_{i}",
                kind="rag",
                role="user",
                content=(
                    f"Document {i} discusses schema architecture, UUIDv7, and index optimization."
                ),
                priority=Priority.P3,
                strategy=Drop(),
                score=i / 100.0,
            )
        )

    # Add 98 conversation history turns
    for i in range(98):
        role = "assistant" if i % 2 == 0 else "user"
        blocks.append(
            ContextBlock(
                id=f"history_turn_{i}",
                kind="history",
                role=role,
                content=f"Historical message turn {i}: confirming parameters and requirements.",
                priority=Priority.P4,
                strategy=Drop(),
                seq=i,
            )
        )

    return blocks


def test_assembly_latency_200_blocks(benchmark, sample_blocks_200: list[ContextBlock]) -> None:  # type: ignore[no-untyped-def]
    model = ModelProfile(
        context_window=32_000,
        max_output_tokens=4_000,
        per_message_overhead=4,
        safety_margin_ratio=0.03,
    )
    tokenizer = CachedTokenizer(HeuristicTokenizer(chars_per_token=4.0), maxsize=2000)
    assembler = Assembler(model=model, tokenizer=tokenizer)

    # Warm up cache so token counts are cached
    assembler.assemble(sample_blocks_200)

    # Benchmark assembly speed
    result = benchmark(assembler.assemble, sample_blocks_200)
    assert result.report.total_tokens <= result.report.input_budget
