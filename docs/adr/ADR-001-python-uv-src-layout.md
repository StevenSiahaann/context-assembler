# ADR-001: Python 3.11+, uv Packaging, and src Layout

## Status
Accepted

## Context
Context Assembler Engine requires high-performance token counting, deterministic mathematical allocation, and seamless integration with the modern AI engineering ecosystem (LangGraph, OpenAI, Anthropic, LiteLLM).

## Decision
1. We target **Python ≥ 3.11** (with 3.12 default) to leverage modern union syntax (`|`), `match` statement ergonomics, and speed optimizations.
2. We manage environments and packaging using **uv** for sub-second dependency resolution.
3. We adopt a strict **src-layout** (`src/context_assembler`) to prevent accidental imports of uninstalled local packages during testing.
4. Core dependencies are kept at **zero mandatory dependencies**. Specialized tokenizers (`tiktoken`) and telemetry (`opentelemetry-api`) are packaged as optional extras.

## Consequences
- Fast CI pipeline (< 2s test suite).
- Clean dependency boundary for external enterprise adoption.
