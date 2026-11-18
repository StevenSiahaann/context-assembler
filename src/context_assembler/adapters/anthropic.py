"""Anthropic Messages API format adapter with multi-breakpoint prompt caching."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, cast

from context_assembler.domain.block import ContextBlock


def _normalize_tool_input(tool_input: Mapping[str, Any] | str | None) -> dict[str, Any]:
    if tool_input is None:
        return {}
    if isinstance(tool_input, Mapping):
        return dict(tool_input)
    try:
        parsed: Any = json.loads(tool_input)
        if isinstance(parsed, dict):
            return dict(cast(dict[str, Any], parsed))
        return {"value": parsed}
    except json.JSONDecodeError:
        return {"raw": tool_input}


def _attach_ephemeral_cache_control(msg: dict[str, Any]) -> None:
    """Attaches Anthropic ephemeral cache_control to a message dict in place."""
    content: Any = msg.get("content")
    if isinstance(content, str):
        msg["content"] = [
            {
                "type": "text",
                "text": content,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    elif isinstance(content, list) and content:
        parts = cast(list[dict[str, Any]], content)
        last_part = parts[-1]
        last_part["cache_control"] = {"type": "ephemeral"}


def to_anthropic_payload(
    blocks: Sequence[ContextBlock],
    enable_cache_control: bool = True,
    mark_history_breakpoint: bool = True,
    max_cache_breakpoints: int = 4,
) -> dict[str, Any]:
    """Converts blocks into Anthropic's top-level system parameter and messages array.

    Places ephemeral cache_control breakpoints on:
    1. System blocks marked `cacheable=True`
    2. Message blocks explicitly marked `cacheable=True`
    3. The final conversation `history` block before turn-local RAG/user query
       (when `mark_history_breakpoint=True`), up to `max_cache_breakpoints` (4).
    """
    system_parts: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []
    message_kinds: list[str] = []
    message_cacheable: list[bool] = []
    breakpoints_used = 0

    for b in blocks:
        if b.role in ("system", "developer"):
            part: dict[str, Any] = {"type": "text", "text": b.content}
            if enable_cache_control and b.cacheable and breakpoints_used < max_cache_breakpoints:
                part["cache_control"] = {"type": "ephemeral"}
                breakpoints_used += 1
            system_parts.append(part)
        elif b.kind == "tool_call" and b.group_id:
            content_parts: list[dict[str, Any]] = []
            if b.content:
                content_parts.append({"type": "text", "text": b.content})
            content_parts.append(
                {
                    "type": "tool_use",
                    "id": b.group_id,
                    "name": b.tool_name or b.source or "tool",
                    "input": _normalize_tool_input(b.tool_input),
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": content_parts,
                }
            )
            message_kinds.append(b.kind)
            message_cacheable.append(b.cacheable)
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
            message_kinds.append(b.kind)
            message_cacheable.append(b.cacheable)
        else:
            messages.append(
                {
                    "role": b.role,
                    "content": b.content,
                }
            )
            message_kinds.append(b.kind)
            message_cacheable.append(b.cacheable)

    if enable_cache_control and messages and breakpoints_used < max_cache_breakpoints:
        # Mark explicitly cacheable non-system messages first
        marked_indices: set[int] = set()
        for idx, is_cacheable in enumerate(message_cacheable):
            if is_cacheable and breakpoints_used < max_cache_breakpoints:
                _attach_ephemeral_cache_control(messages[idx])
                marked_indices.add(idx)
                breakpoints_used += 1

        # Place breakpoint on the last stable history message before dynamic RAG/active user turn
        if mark_history_breakpoint and breakpoints_used < max_cache_breakpoints:
            history_indices = [i for i, k in enumerate(message_kinds) if k == "history"]
            if history_indices:
                last_hist_idx = history_indices[-1]
                if last_hist_idx not in marked_indices:
                    _attach_ephemeral_cache_control(messages[last_hist_idx])
                    breakpoints_used += 1

    result: dict[str, Any] = {"messages": messages}
    if system_parts:
        result["system"] = system_parts

    return result
