# ADR-003: Tokenizer Protocol and Safety Margins

## Status
Accepted

## Context
Different LLM providers utilize distinct BPE tokenizer vocabularies (e.g. OpenAI `o200k_base`, `cl100k_base`, Llama 3 `tiktoken`, Anthropic proprietary). Furthermore, Anthropic does not provide an official local client-side tokenizer library.

## Decision
1. Define a `Tokenizer` `Protocol` with `count(text)` and `count_lines(lines)`.
2. Wrap inner tokenizers with `CachedTokenizer` using bounded LRU and 16-byte blake2b key hashing to avoid storing massive strings in memory.
3. Configure `safety_margin_ratio` (default 3% for OpenAI, 8% for Anthropic) to absorb local estimation divergence before requests reach the provider.
4. Pass `disallowed_special=()` in `tiktoken` to prevent untrusted user data containing `<|endoftext|>` from crashing execution.

## Consequences
- Zero-downtime tolerance for tokenizer drift.
- Fast caching with safe memory boundaries.

