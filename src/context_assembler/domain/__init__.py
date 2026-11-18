"""Domain types, invariants, and policies."""

from context_assembler.domain.block import (
    Alternatives,
    BlockKind,
    ContextBlock,
    Drop,
    HeadTail,
    MiddleOut,
    Never,
    Priority,
    ReductionStrategy,
    Role,
    Summarize,
    TokenizedBlock,
)
from context_assembler.domain.budget import ModelProfile, SlotPolicy
from context_assembler.domain.errors import (
    AssemblyError,
    InvalidBlockError,
    OrphanToolResultError,
    P0OverflowError,
    StructuralValidityError,
)

__all__ = [
    "Alternatives",
    "AssemblyError",
    "BlockKind",
    "ContextBlock",
    "Drop",
    "HeadTail",
    "InvalidBlockError",
    "MiddleOut",
    "ModelProfile",
    "Never",
    "OrphanToolResultError",
    "P0OverflowError",
    "Priority",
    "ReductionStrategy",
    "Role",
    "SlotPolicy",
    "StructuralValidityError",
    "Summarize",
    "TokenizedBlock",
]

