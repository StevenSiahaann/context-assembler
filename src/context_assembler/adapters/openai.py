"""OpenAI Chat Completion format adapter."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from context_assembler.domain.block import ContextBlock


def _serialize_tool_arguments(tool_input: Mapping[str, Any] | str | None) -> str:
    if tool_input is None:
        return "{}"
    if isinstance(tool_input, str):
        return tool_input
    return json.dumps(dict(tool_input), sort_keys=True, separators=(",", ":"))


def to_openai_messages(
    blocks: Sequence[ContextBlock],
    system_role: str = "system",
) -> list[dict[str, Any]]:
    """Converts assembled blocks into OpenAI chat messages format."""
    messages: list[dict[str, Any]] = []

    for b in blocks:
        role = system_role if b.role in ("system", "developer") else b.role

        if b.kind == "tool_call" and b.group_id:
            messages.append(
                {
                    "role": "assistant",
                    "content": b.content,
                    "tool_calls": [
                        {
                            "id": b.group_id,
                            "type": "function",
                            "function": {
                                "name": b.tool_name or b.source or "tool",
                                "arguments": _serialize_tool_arguments(b.tool_input),
                            },
                        }
                    ],
                }
            )
        elif b.kind == "tool_result" and b.group_id:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": b.group_id,
                    "content": b.content,
                }
            )
        else:
            messages.append(
                {
                    "role": role,
                    "content": b.content,
                }
            )

    return messages
