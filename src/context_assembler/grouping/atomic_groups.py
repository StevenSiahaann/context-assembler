"""Atomic grouping validation for tool call/result pairs."""

from __future__ import annotations

from collections.abc import Sequence

from context_assembler.domain.block import ContextBlock
from context_assembler.domain.errors import OrphanToolResultError


def validate_atomic_groups(blocks: Sequence[ContextBlock]) -> None:
    """Ensures tool calls and tool results are mutually paired by group_id.

    Providers (OpenAI, Anthropic) return HTTP 400 errors if a tool response
    message appears without an assistant tool_call, or if an assistant tool_call
    is not followed by its corresponding tool_result.
    """
    call_groups: set[str] = set()
    result_groups: set[str] = set()

    for b in blocks:
        if b.kind == "tool_call" and b.group_id:
            call_groups.add(b.group_id)
        elif b.kind == "tool_result" and b.group_id:
            result_groups.add(b.group_id)

    for b in blocks:
        if b.kind == "tool_result" and b.group_id and b.group_id not in call_groups:
            raise OrphanToolResultError(b.group_id)
        if b.kind == "tool_call" and b.group_id and b.group_id not in result_groups:
            raise OrphanToolResultError(b.group_id)
