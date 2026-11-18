"""Unit tests for OpenAI and Anthropic provider adapters."""

from context_assembler.adapters.anthropic import to_anthropic_payload
from context_assembler.adapters.openai import to_openai_messages
from context_assembler.domain.block import (
    ContextBlock,
    Never,
    Priority,
)


def test_openai_adapter_formatting() -> None:
    blocks = [
        ContextBlock(
            id="sys_1",
            kind="system",
            role="system",
            content="You are a data architect.",
            priority=Priority.P0,
            strategy=Never(),
        ),
        ContextBlock(
            id="user_1",
            kind="user",
            role="user",
            content="Show table DDL",
            priority=Priority.P1,
            strategy=Never(),
        ),
        ContextBlock(
            id="tool_res_1",
            kind="tool_result",
            role="tool",
            content="CREATE TABLE users (id UUID PRIMARY KEY);",
            priority=Priority.P2,
            strategy=Never(),
            group_id="call_ddl_123",
        ),
    ]

    messages = to_openai_messages(blocks, system_role="developer")

    assert len(messages) == 3
    assert messages[0] == {"role": "developer", "content": "You are a data architect."}
    assert messages[1] == {"role": "user", "content": "Show table DDL"}
    assert messages[2] == {
        "role": "tool",
        "tool_call_id": "call_ddl_123",
        "content": "CREATE TABLE users (id UUID PRIMARY KEY);",
    }


def test_anthropic_adapter_separates_system_and_adds_cache_control() -> None:
    blocks = [
        ContextBlock(
            id="sys_1",
            kind="system",
            role="system",
            content="You are Claude.",
            priority=Priority.P0,
            strategy=Never(),
            cacheable=True,
        ),
        ContextBlock(
            id="user_1",
            kind="user",
            role="user",
            content="Hello",
            priority=Priority.P1,
            strategy=Never(),
        ),
    ]

    payload = to_anthropic_payload(blocks, enable_cache_control=True)

    # System must be top-level list
    assert "system" in payload
    assert payload["system"] == [
        {"type": "text", "text": "You are Claude.", "cache_control": {"type": "ephemeral"}}
    ]

    # Messages array must only contain the user turn
    assert payload["messages"] == [{"role": "user", "content": "Hello"}]
