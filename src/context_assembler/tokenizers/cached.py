"""Thread-safe bounded LRU cache for tokenizer counts."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from threading import Lock

from context_assembler.tokenizers.base import Tokenizer


class CachedTokenizer:
    """Wraps a Tokenizer with a thread-safe, bounded LRU cache.

    Crucially, cache keys use a 16-byte blake2b digest rather than storing
    the raw string directly. This prevents multi-megabyte tool outputs or log
    dumps from exhausting process memory while sitting in the LRU cache.
    """

    def __init__(self, inner: Tokenizer, maxsize: int = 10_000) -> None:
        if maxsize <= 0:
            raise ValueError(f"maxsize must be positive, got {maxsize}")
        self._inner = inner
        self._maxsize = maxsize
        self._cache: OrderedDict[bytes, int] = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    @property
    def name(self) -> str:
        return f"cached:{self._inner.name}"

    @property
    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._cache),
                "maxsize": self._maxsize,
            }

    def _hash(self, text: str) -> bytes:
        # 16 bytes gives 128 bits of collision resistance, more than enough for cache keys
        return hashlib.blake2b(text.encode("utf-8", errors="replace"), digest_size=16).digest()

    def count(self, text: str) -> int:
        key = self._hash(text)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1

        val = self._inner.count(text)

        with self._lock:
            self._cache[key] = val
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

        return val

    def count_lines(self, lines: list[str]) -> list[int]:
        # Delegate individual lines so frequent repeating lines hit cache
        return [self.count(line) for line in lines]

