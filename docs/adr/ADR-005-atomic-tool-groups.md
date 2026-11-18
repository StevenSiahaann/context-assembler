# ADR-005: Atomic Tool Call and Result Groups

## Status
Accepted

## Context
When an LLM executes a tool, the provider records an `assistant` message containing a `tool_calls` payload, followed by a `tool` role message returning the output. If a context manager evicts the tool call while retaining the result (or vice versa), provider APIs return fatal HTTP 400 bad request errors.

## Decision
1. Invariant: `tool_call` and its associated `tool_result` share an identical `group_id`.
2. Atomic eviction: Reduction algorithms (e.g. `reduce_anchored_window`) evict or retain all members of a `group_id` together.
3. If tool result payload is excessively large, apply `HeadTail` reduction to the content *inside* the result rather than dropping the pair.

## Consequences
- Guaranteed API protocol validity under context trimming.
- Agent loops remain coherent because tool execution history is never partially orphaned.

