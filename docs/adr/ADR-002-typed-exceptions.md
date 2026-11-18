# ADR-002: Typed Exception Hierarchy for Domain Errors

## Status
Accepted

## Context
While functional languages and some TypeScript frameworks mandate `Result<T, E>` monads, standard Python libraries and async agent harnesses rely on structured Python exception hierarchies for recovery routing.

## Decision
We implement a typed exception hierarchy rooted in `AssemblyError`:
- `P0OverflowError(required, available)`: Raised when non-negotiable instructions exceed budget.
- `InvalidBlockError(block_id, reason)`: Domain validation failure.
- `OrphanToolResultError(group_id)`: Unpaired tool result violation.
- `StructuralValidityError(rule, detail)`: Message sequencing violation.

Callers can selectively intercept `P0OverflowError` to truncate system prompts or alert engineers, while letting programming errors propagate cleanly.

## Consequences
- Idiomatic Python usage without monadic unwrap boilerplate.
- Rich error attributes for automated retry and telemetry logging.

