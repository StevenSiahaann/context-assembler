"""Context Assembler Engine.

Deterministic, priority-based token budgeting and context assembly engine for LLMs.
"""

from context_assembler.adapters.anthropic import to_anthropic_payload
from context_assembler.adapters.openai import to_openai_messages
from context_assembler.assembler import AssembledContext, Assembler
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
from context_assembler.observability.report import (
    AssemblyReport,
    BlockDecision,
    SlotUsage,
)
from context_assembler.tokenizers.base import Tokenizer
from context_assembler.tokenizers.cached import CachedTokenizer
from context_assembler.tokenizers.heuristic import HeuristicTokenizer
from context_assembler.tokenizers.tiktoken_ import TiktokenTokenizer

__all__ = [
    "Alternatives",
    "AssembledContext",
    "Assembler",
    "AssemblyError",
    "AssemblyReport",
    "BlockDecision",
    "BlockKind",
    "CachedTokenizer",
    "ContextBlock",
    "Drop",
    "HeadTail",
    "HeuristicTokenizer",
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
    "SlotUsage",
    "StructuralValidityError",
    "Summarize",
    "TiktokenTokenizer",
    "TokenizedBlock",
    "Tokenizer",
    "to_anthropic_payload",
    "to_openai_messages",
]

__version__ = "0.1.0"
