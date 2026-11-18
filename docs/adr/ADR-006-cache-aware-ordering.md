# ADR-006: Cache-Aware Message Ordering and Prefix Stabilization

## Status
Accepted

## Context
Modern LLM APIs (Anthropic, OpenAI, DeepSeek, Google Gemini) offer significant cost reductions (50% to 90%) when prompt prefixes match cached KV states. Inserting dynamic RAG chunks or timestamps early in the prompt invalidates the cache for all subsequent tokens.

## Decision
1. **Static Prefix Segment**: System prompts, tool definitions, and pinned core memory are placed at the absolute top of the prompt payload, serialized with deterministic key ordering.
2. **Chronological History**: Prior conversation turns follow the static prefix.
3. **Turn-Local Dynamic Context**: Dynamic RAG documents and active user prompts are placed at the very end of the prompt payload.
4. **Telemetry**: Compute and report `prefix_hash` to detect accidental cache invalidations across requests.

## Consequences
- Maximizes prompt cache hit rates across multi-turn agent sessions.
- Keeps RAG context near the final user question, mitigating "lost-in-the-middle" attention degradation.
