# Context Assembler Engine

[![CI](https://github.com/stevengianmarg/context-assembler/actions/workflows/ci.yml/badge.svg)](https://github.com/stevengianmarg/context-assembler)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](pyproject.toml)
[![Type Checked](https://img.shields.io/badge/pyright-strict-green.svg)](pyproject.toml)

> **Deterministic, priority-based token budgeting and context assembly engine for LLM applications and autonomous agents.**

---

## The Problem

Most LLM frameworks handle context limits using **naive sliding windows** (dropping earlier turns) or **blind string truncation**. In real-world enterprise applications, this causes:

1. **Context Window Exceeded (HTTP 400)** errors when tool outputs or RAG chunks spike unexpectedly.
2. **Broken Tool Pairing**, where an assistant tool call is retained but its tool output is evicted.
3. **Severe Cache Thrashing**, where interleaving dynamic RAG chunks throughout the prompt destroys KV prefix cache hit rates.
4. **Prompt Delimiter Injection**, where untrusted web or tool data breaks structural boundaries.

---

## Architectural Guarantees

* **Zero Overflow Invariant:** Mathematically guarantees prompt token counts never exceed available input budget (verified via Hypothesis property testing).
* **Priority-Based Water-Filling:** Fairly partitions dynamic capacity across competing slots (`rag`, `history`, `tool_result`, `memory`) with guaranteed floors and proportional surplus sharing.
* **Atomic Group Preservation:** Tool calls and their corresponding execution outputs are never separated during eviction.
* **Cache-Aware Layout:** Groups static instructions and tool definitions at the prefix to maximize provider prompt caching discounts (up to 90% cost reduction).
* **Sub-Millisecond Speed:** Assembles 200 blocks in **~0.40 ms** (12x faster than the 5ms latency budget).

---

## Quickstart

### Installation

```bash
# Core package (zero dependencies)
pip install context-assembler

# With official OpenAI tiktoken support
pip install 'context-assembler[openai]'
```

### Basic Usage

```python
from context_assembler import (
    Assembler,
    ContextBlock,
    HeadTail,
    Drop,
    ModelProfile,
    Never,
    Priority,
    SlotPolicy,
    TiktokenTokenizer,
    CachedTokenizer,
    to_openai_messages,
)

# 1. Define model profile and slot capacities
model = ModelProfile(
    context_window=128_000,
    max_output_tokens=4_000,
    per_message_overhead=4,
    safety_margin_ratio=0.03,
)

tokenizer = CachedTokenizer(TiktokenTokenizer("o200k_base"), maxsize=10_000)

assembler = Assembler(
    model=model,
    tokenizer=tokenizer,
    slots=[
        SlotPolicy("tool_result", min_tokens=1_000, max_tokens=15_000, weight=3),
        SlotPolicy("rag", min_tokens=1_000, max_tokens=25_000, weight=4),
        SlotPolicy("history", min_tokens=2_000, max_tokens=30_000, weight=2),
    ],
)

# 2. Collect context blocks from your agent loop
blocks = [
    ContextBlock(
        id="system_prompt",
        kind="system",
        role="system",
        content="You are an autonomous PostgreSQL database architect.",
        priority=Priority.P0,
        strategy=Never(),
        cacheable=True,
    ),
    ContextBlock(
        id="user_intent",
        kind="user",
        role="user",
        content="Diagnose why queries on order_events are running slowly.",
        priority=Priority.P1,
        strategy=HeadTail(0.5),
    ),
    ContextBlock(
        id="query_log_dump",
        kind="tool_result",
        role="tool",
        content="[... 10,000 lines of EXPLAIN ANALYZE logs ...]",
        priority=Priority.P2,
        strategy=HeadTail(0.5),
        group_id="call_explain_analyze",
    ),
]

# 3. Assemble and dispatch
result = assembler.assemble(blocks)

print(f"Assembly completed in {result.report.duration_ms} ms")
print(f"Utilization: {result.report.utilization * 100:.1f}%")
print(f"Tokens: {result.report.total_tokens} / {result.report.input_budget}")

messages = to_openai_messages(result.blocks)
```

---

## Benchmark Results

Benchmarked on Apple Silicon (M-series) using `pytest-benchmark` over 2,000 iterations:

| Scenario | Candidate Size | Resulting Latency | Status |
| :--- | :--- | :--- | :--- |
| **Full Assembly (200 Blocks)** | 100 RAG Docs + 98 History Turns + System & User | **0.40 ms** | PASS (< 5.0 ms target) |

---

## Architecture Decision Records (ADRs)

Key architectural choices are documented under `docs/adr/`:
* [ADR-001: Python 3.11+, uv Packaging, and src Layout](docs/adr/ADR-001-python-uv-src-layout.md)
* [ADR-002: Typed Exception Hierarchy for Domain Errors](docs/adr/ADR-002-typed-exceptions.md)
* [ADR-003: Tokenizer Protocol and Safety Margins](docs/adr/ADR-003-tokenizer-protocol-and-margins.md)
* [ADR-004: Priority-Based Water-Filling Budget Allocation](docs/adr/ADR-004-water-filling-allocation.md)
* [ADR-005: Atomic Tool Call and Result Groups](docs/adr/ADR-005-atomic-tool-groups.md)
* [ADR-006: Cache-Aware Message Ordering and Prefix Stabilization](docs/adr/ADR-006-cache-aware-ordering.md)

---

## License

Apache-2.0
