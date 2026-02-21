"""Tests for HybridReranker, AdaptiveBatcher, CircuitBreaker."""

import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from hybrid_engine import AdaptiveBatcher, CircuitBreaker, HybridReranker


# ---------------------------------------------------------------------------
# AdaptiveBatcher
# ---------------------------------------------------------------------------


class TestAdaptiveBatcher:
    def test_initial_size(self):
        b = AdaptiveBatcher(initial_size=32)
        assert b.size == 32

    def test_shrink_on_oom(self):
        b = AdaptiveBatcher(initial_size=32, min_size=4)
        b.adjust(success=False)
        assert b.size == 16
        b.adjust(success=False)
        assert b.size == 8
        b.adjust(success=False)
        assert b.size == 4
        # Should not go below min
        b.adjust(success=False)
        assert b.size == 4

    def test_grow_after_successes(self):
        b = AdaptiveBatcher(initial_size=16, max_size=64)
        for _ in range(10):
            b.adjust(success=True)
        # Should have grown after 10 successes
        assert b.size > 16

    def test_does_not_exceed_max(self):
        b = AdaptiveBatcher(initial_size=60, max_size=64)
        for _ in range(100):
            b.adjust(success=True)
        assert b.size <= 64


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker()
        assert cb.state == "closed"
        assert cb.can_attempt() is True

    def test_opens_after_failures(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == "open"
        assert cb.can_attempt() is False

    def test_closes_on_success(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == "open"

        # Simulate timeout by manipulating last_failure_time
        cb.last_failure_time = 0
        # Now it should be half-open
        assert cb.can_attempt() is True
        assert cb.state == "half_open"

        cb.record_success()
        assert cb.state == "closed"
        assert cb.failure_count == 0

    def test_stays_below_threshold(self):
        cb = CircuitBreaker(failure_threshold=5)
        for _ in range(4):
            cb.record_failure()
        assert cb.state == "closed"
        assert cb.can_attempt() is True


# ---------------------------------------------------------------------------
# HybridReranker
# ---------------------------------------------------------------------------


class TestHybridReranker:
    @pytest.mark.asyncio
    async def test_rerank_fast_mode(self, hybrid_reranker, sample_query, sample_documents):
        result = await hybrid_reranker.rerank(
            query=sample_query,
            documents=sample_documents,
            top_k=3,
            mode="fast",
            use_cache=False,
        )

        assert len(result["results"]) == 3
        assert result["mode_used"] == "fast"
        assert result["cache_hit"] is False

        # Scores should be sorted descending
        scores = [r["score"] for r in result["results"]]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_rerank_accurate_mode(self, hybrid_reranker, sample_query, sample_documents):
        result = await hybrid_reranker.rerank(
            query=sample_query,
            documents=sample_documents,
            top_k=5,
            mode="accurate",
            use_cache=False,
        )

        assert len(result["results"]) == 5
        assert result["model_used"] == "accurate"

    @pytest.mark.asyncio
    async def test_rerank_auto_mode(self, hybrid_reranker, sample_query, sample_documents):
        result = await hybrid_reranker.rerank(
            query=sample_query,
            documents=sample_documents,
            top_k=3,
            mode="auto",
            use_cache=False,
        )

        assert len(result["results"]) == 3
        assert result["model_used"] in ("fast", "accurate")

    @pytest.mark.asyncio
    async def test_rerank_top_k_clamps(self, hybrid_reranker, sample_query, sample_documents):
        """top_k larger than doc count should return all docs."""
        result = await hybrid_reranker.rerank(
            query=sample_query,
            documents=sample_documents,
            top_k=100,
            mode="fast",
            use_cache=False,
        )

        assert len(result["results"]) == len(sample_documents)

    @pytest.mark.asyncio
    async def test_rerank_cache_hit(self, hybrid_reranker, sample_query, sample_documents):
        """Second call with same query+docs should be a cache hit."""
        # First call — cache miss
        r1 = await hybrid_reranker.rerank(
            query=sample_query,
            documents=sample_documents,
            top_k=3,
            mode="fast",
            use_cache=True,
        )
        assert r1["cache_hit"] is False

        # Second call — cache hit
        r2 = await hybrid_reranker.rerank(
            query=sample_query,
            documents=sample_documents,
            top_k=3,
            mode="fast",
            use_cache=True,
        )
        assert r2["cache_hit"] is True

    @pytest.mark.asyncio
    async def test_rerank_no_cache(self, hybrid_reranker, sample_query, sample_documents):
        """use_cache=False should bypass cache."""
        await hybrid_reranker.rerank(
            query=sample_query,
            documents=sample_documents,
            top_k=3,
            mode="fast",
            use_cache=True,
        )

        # Even after caching, use_cache=False should recompute
        result = await hybrid_reranker.rerank(
            query=sample_query,
            documents=sample_documents,
            top_k=3,
            mode="fast",
            use_cache=False,
        )
        assert result["cache_hit"] is False

    @pytest.mark.asyncio
    async def test_result_structure(self, hybrid_reranker, sample_query, sample_documents):
        result = await hybrid_reranker.rerank(
            query=sample_query,
            documents=sample_documents,
            top_k=2,
            mode="fast",
            use_cache=False,
        )

        for r in result["results"]:
            assert "document" in r
            assert "score" in r
            assert "model" in r
            assert "confidence" in r
            assert isinstance(r["score"], float)
            assert 0.0 <= r["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_cache_key_deterministic(self, hybrid_reranker):
        """Same query+docs should produce same cache key regardless of doc order."""
        docs = ["doc_b", "doc_a", "doc_c"]
        key1 = hybrid_reranker._generate_cache_key("query", docs)
        key2 = hybrid_reranker._generate_cache_key("query", docs)
        assert key1 == key2

    @pytest.mark.asyncio
    async def test_cache_key_order_independent(self, hybrid_reranker):
        """Different doc orderings should produce same cache key (sorted internally)."""
        key1 = hybrid_reranker._generate_cache_key("q", ["a", "b", "c"])
        key2 = hybrid_reranker._generate_cache_key("q", ["c", "a", "b"])
        assert key1 == key2

    def test_compute_confidence_single(self, hybrid_reranker):
        scores = np.array([0.9])
        conf = hybrid_reranker._compute_confidence(scores)
        assert conf == 1.0

    def test_compute_confidence_high_variance(self, hybrid_reranker):
        scores = np.array([0.95, 0.1, 0.05, 0.02])
        conf = hybrid_reranker._compute_confidence(scores)
        assert conf > 0.5

    def test_compute_confidence_low_variance(self, hybrid_reranker):
        scores = np.array([0.51, 0.50, 0.49, 0.48])
        conf = hybrid_reranker._compute_confidence(scores)
        # Low variance = less confident in ranking
        assert conf < 1.0

    def test_get_stats(self, hybrid_reranker):
        stats = hybrid_reranker.get_stats()
        assert "requests" in stats
        assert "cache_hit_rate" in stats
        assert "current_batch_size" in stats
        assert "circuit_breaker_state" in stats
        assert stats["circuit_breaker_state"] == "closed"
