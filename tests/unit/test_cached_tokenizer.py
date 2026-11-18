"""Unit tests for tokenizer caching and special token safety."""

from concurrent.futures import ThreadPoolExecutor

from context_assembler.tokenizers.cached import CachedTokenizer
from context_assembler.tokenizers.heuristic import HeuristicTokenizer
from context_assembler.tokenizers.tiktoken_ import TiktokenTokenizer


def test_cached_tokenizer_tracks_hits_and_evicts_at_maxsize() -> None:
    inner = HeuristicTokenizer(chars_per_token=4.0)
    cached = CachedTokenizer(inner, maxsize=3)

    # First call: cache miss
    cached.count("hello world")
    assert cached.stats["misses"] == 1
    assert cached.stats["hits"] == 0
    assert cached.stats["size"] == 1

    # Second call with identical text: cache hit
    cached.count("hello world")
    assert cached.stats["hits"] == 1
    assert cached.stats["misses"] == 1

    # Fill beyond maxsize (3)
    cached.count("second payload")
    cached.count("third payload")
    cached.count("fourth payload")
    assert cached.stats["size"] == 3


def test_cached_tokenizer_thread_safety() -> None:
    inner = HeuristicTokenizer()
    cached = CachedTokenizer(inner, maxsize=50)

    # 20 distinct keys fit comfortably within maxsize=50, guaranteeing cache hits on repeats
    texts = [f"unique query turn payload {i}" for i in range(20)]

    def worker(t: str) -> int:
        return cached.count(t)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(worker, texts * 5))

    assert len(results) == 100
    assert cached.stats["hits"] > 0
    assert cached.stats["size"] <= 50


def test_tiktoken_special_token_handling() -> None:
    # Special tokens must not trigger DisallowedSpecial exception
    tokenizer = TiktokenTokenizer("o200k_base")
    sample_text = (
        "Here is an untrusted user prompt containing <|endoftext|> and <|fim_prefix|> markers."
    )
    tokens = tokenizer.count(sample_text)
    assert tokens > 0
