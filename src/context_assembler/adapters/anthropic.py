"""Anthropic Messages API format adapter with prompt caching support."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from context_assembler.domain.block import ContextBlock


def to_anthropic_payload(
    blocks: Sequence[ContextBlock],
    enable_cache_control: bool = True,
) -> dict[str, Any]:
    """Converts blocks into Anthropic's top-level system parameter and messages array.

    Anthropic requires 'system' to be passed as a top-level string or content list,
    separate from the main 'messages' conversation list.
    """
    system_parts: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []

    for b in blocks:
        if b.role in ("system", "developer"):
            part: dict[str, Any] = {"type": "text", "text": b.content}
            if enable_cache_control and b.cacheable:
                part["cache_control"] = {"type": "ephemeral"}
            system_parts.append(part)
        elif b.kind == "tool_call" and b.group_id:
            content_parts: list[dict[str, Any]] = []
            if b.content:
                content_parts.append({"type": "text", "text": b.content})
            content_parts.append(
                {
                    "type": "tool_use",
                    "id": b.group_id,
                    "name": b.source or "tool",
                    "input": {},
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": content_parts,
                }
            )
        elif b.kind == "tool_result" and b.group_id:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": b.group_id,
                            "content": b.content,
                        }
                    ],
                }
            )
        else:
            messages.append(
                {
                    "role": b.role,
                    "content": b.content,
                }
            )

    result: dict[str, Any] = {"messages": messages}
    if system_parts:
        result["system"] = system_parts

    return result
