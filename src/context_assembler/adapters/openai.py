"""OpenAI Chat Completion format adapter."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from context_assembler.domain.block import ContextBlock


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
                                "name": b.source or "tool",
                                "arguments": "{}",
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
