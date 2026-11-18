# ADR-004: Priority-Based Water-Filling Budget Allocation

## Status
Accepted

## Context
Naive prompt truncation either uses sliding window (which starves RAG retrieval when conversation grows) or greedy knapsack (which starves history when retrieved documents are large).

## Decision
We implement a two-phase **water-filling** allocation algorithm across dynamic slots:
1. **Floor Phase**: Allocate guaranteed minimum tokens (`min_tokens`) to each slot in strict priority order.
2. **Proportional Surplus Phase**: Distribute remaining tokens proportionally according to slot `weight`, bounded by `max_tokens` and actual demand.
3. **Rounding Remainder**: When integer division rounds down, distribute remaining single tokens to active slots in priority order.

## Consequences
- Guaranteed baseline capacity for both RAG context and conversational history.
- Mathematically bounded execution terminating in $\le S$ iterations (where $S$ is number of slots $\le 10$).
