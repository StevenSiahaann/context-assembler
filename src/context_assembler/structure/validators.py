"""Verification checks against provider API invariant rules."""

from __future__ import annotations

from collections.abc import Sequence

from context_assembler.domain.block import TokenizedBlock
from context_assembler.domain.errors import StructuralValidityError


def validate_assembled_structure(
    blocks: Sequence[TokenizedBlock],
    requires_alternation: bool = False,
) -> None:
    """Verifies that an assembled block sequence is guaranteed valid for provider dispatch."""
    if not blocks:
        return

    chat_blocks = [tb for tb in blocks if tb.block.role not in ("system", "developer")]
    if not chat_blocks:
        return

    # Invariant: First chat turn must not be an assistant turn
    if chat_blocks[0].block.role == "assistant":
        raise StructuralValidityError(
            rule="first_chat_turn_must_be_user",
            detail="Chat history begins with an assistant message instead of user.",
        )

    # Invariant: Consecutive roles if alternation is mandated
    if requires_alternation:
        for i in range(len(chat_blocks) - 1):
            curr_role = chat_blocks[i].block.role
            next_role = chat_blocks[i + 1].block.role
            if curr_role == next_role and curr_role in ("user", "assistant"):
                raise StructuralValidityError(
                    rule="strict_alternation_required",
                    detail=(
                        f"Consecutive messages found with role {curr_role!r} "
                        f"at indices {i} and {i + 1}"
                    ),
                )
