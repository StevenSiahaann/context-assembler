"""Structural normalization to satisfy model provider protocol requirements."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from context_assembler.domain.block import ContextBlock, Priority, TokenizedBlock
from context_assembler.tokenizers.base import Tokenizer


def normalize_message_structure(
    blocks: Sequence[TokenizedBlock],
    requires_alternation: bool = False,
    tokenizer: Tokenizer | None = None,
) -> list[TokenizedBlock]:
    """Ensures message sequence adheres to provider schema rules.

    Rules applied:
    1. Filter out empty message payloads.
    2. Ensure conversation begins with a 'user' message (after any system/developer prompts).
       If an assistant message was stranded at the head after history truncation,
       we prepend a synthetic user continuity stub.
    3. If requires_alternation is True (Anthropic/Gemini), merge consecutive
       identical role turns into a single unified turn separated by double newlines.
    """
    if not blocks:
        return []

    # 1. Filter out empty content blocks
    non_empty = [tb for tb in blocks if tb.block.content.strip()]
    if not non_empty:
        return []

    # Separate system/developer instructions from conversation flow
    system_blocks = [tb for tb in non_empty if tb.block.role in ("system", "developer")]
    chat_blocks = [tb for tb in non_empty if tb.block.role not in ("system", "developer")]

    if not chat_blocks:
        return list(system_blocks)

    # 2. Ensure first chat turn is from 'user'
    first_chat = chat_blocks[0]
    if first_chat.block.role in ("assistant", "tool"):
        stub_block = ContextBlock(
            id="__synthetic_user_continuity_turn__",
            kind="user",
            role="user",
            content="[Conversation resumed with earlier context truncated]",
            priority=Priority.P1,
            strategy=first_chat.block.strategy,
        )
        cost = tokenizer.count(stub_block.content) if tokenizer else 12
        chat_blocks.insert(0, TokenizedBlock(stub_block, cost))

    # 3. Handle alternation if required by model provider
    if not requires_alternation:
        return [*system_blocks, *chat_blocks]

    merged_chat: list[TokenizedBlock] = []
    current_acc: TokenizedBlock | None = None

    for tb in chat_blocks:
        if current_acc is None:
            current_acc = tb
            continue

        if tb.block.role == current_acc.block.role and not (
            tb.block.group_id or current_acc.block.group_id
        ):
            # Merge same role content
            combined_content = f"{current_acc.block.content}\n\n{tb.block.content}"
            new_tokens = (
                tokenizer.count(combined_content)
                if tokenizer
                else current_acc.tokens + tb.tokens + 2
            )
            merged_block = replace(current_acc.block, content=combined_content)
            current_acc = TokenizedBlock(merged_block, new_tokens)
        else:
            merged_chat.append(current_acc)
            current_acc = tb

    if current_acc is not None:
        merged_chat.append(current_acc)

    return [*system_blocks, *merged_chat]
