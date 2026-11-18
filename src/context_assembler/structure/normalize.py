"""Structural normalization to satisfy model provider protocol requirements."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from context_assembler.domain.block import TokenizedBlock
from context_assembler.tokenizers.base import Tokenizer


def normalize_message_structure(
    blocks: Sequence[TokenizedBlock],
    requires_alternation: bool = False,
    tokenizer: Tokenizer | None = None,
) -> tuple[list[TokenizedBlock], list[TokenizedBlock]]:
    """Ensures message sequence adheres to provider schema rules.

    Rules applied:
    1. Filter out empty message payloads.
    2. Ensure conversation begins with a 'user' message (after any system/developer prompts).
       Leading orphaned assistant/tool turns (and their atomic group members) are evicted
       so token usage only decreases and never overruns the budget.
    3. If requires_alternation is True (Anthropic/Gemini), merge consecutive
       identical role turns into a single unified turn separated by a newline.

    Returns:
        tuple of (normalized_blocks, structurally_dropped_blocks)
    """
    if not blocks:
        return [], []

    dropped: list[TokenizedBlock] = []
    non_empty: list[TokenizedBlock] = []
    for tb in blocks:
        if tb.block.content.strip():
            non_empty.append(tb)
        else:
            dropped.append(tb)

    if not non_empty:
        return [], dropped

    system_blocks = [tb for tb in non_empty if tb.block.role in ("system", "developer")]
    chat_blocks = [tb for tb in non_empty if tb.block.role not in ("system", "developer")]

    if not chat_blocks:
        return list(system_blocks), dropped

    # 2. Evict leading non-user turns (and any linked atomic groups) until a user turn leads
    evicted_groups: set[str] = set()
    idx = 0
    while idx < len(chat_blocks) and chat_blocks[idx].block.role in ("assistant", "tool"):
        lead = chat_blocks[idx]
        dropped.append(lead)
        if lead.block.group_id:
            evicted_groups.add(lead.block.group_id)
        idx += 1

    remaining_chat: list[TokenizedBlock] = []
    for tb in chat_blocks[idx:]:
        if tb.block.group_id and tb.block.group_id in evicted_groups:
            dropped.append(tb)
        else:
            remaining_chat.append(tb)

    chat_blocks = remaining_chat
    if not chat_blocks:
        return list(system_blocks), dropped

    # 3. Handle alternation if required by model provider
    if not requires_alternation:
        return [*system_blocks, *chat_blocks], dropped

    merged_chat: list[TokenizedBlock] = []
    current_acc: TokenizedBlock | None = None

    for tb in chat_blocks:
        if current_acc is None:
            current_acc = tb
            continue

        if (
            tb.block.role == current_acc.block.role
            and tb.block.role in ("user", "assistant")
            and not (tb.block.group_id or current_acc.block.group_id)
            and not (
                (tb.block.priority == 0 or tb.block.pinned)
                and (current_acc.block.priority == 0 or current_acc.block.pinned)
            )
        ):
            combined_content = f"{current_acc.block.content}\n{tb.block.content}"
            new_tokens = (
                tokenizer.count(combined_content)
                if tokenizer
                else current_acc.tokens + tb.tokens + 1
            )
            base_block = (
                tb.block
                if (tb.block.pinned or tb.block.priority < current_acc.block.priority)
                else current_acc.block
            )
            merged_trust = (
                "untrusted"
                if (current_acc.block.trust == "untrusted" or tb.block.trust == "untrusted")
                else "trusted"
            )
            merged_block = replace(base_block, content=combined_content, trust=merged_trust)
            current_acc = TokenizedBlock(merged_block, new_tokens)
        else:
            merged_chat.append(current_acc)
            current_acc = tb

    if current_acc is not None:
        merged_chat.append(current_acc)

    return [*system_blocks, *merged_chat], dropped
