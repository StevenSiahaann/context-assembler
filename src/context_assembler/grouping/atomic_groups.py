"""Atomic grouping validation for tool call/result pairs."""

from __future__ import annotations

from collections.abc import Sequence

from context_assembler.domain.block import ContextBlock
from context_assembler.domain.errors import OrphanToolResultError


def validate_atomic_groups(blocks: Sequence[ContextBlock]) -> None:
    """Ensures tool results have matching tool calls.

    Providers (OpenAI, Anthropic) will return 400 errors if a tool response
    message appears without an assistant tool_call earlier in the conversation.
    """
    call_groups: set[str] = set()

    for b in blocks:
        if b.kind == "tool_call" and b.group_id:
            call_groups.add(b.group_id)

    for b in blocks:
        if b.kind == "tool_result" and b.group_id and b.group_id not in call_groups:
            raise OrphanToolResultError(b.group_id)
