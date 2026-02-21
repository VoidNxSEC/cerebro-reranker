"""Tests for CerebroReranker client (migration-aware)."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from cerebro.reranker_client import CerebroReranker, RerankResult


def _make_result(backend="local", docs=None, scores=None):
    return RerankResult(
        ranked_docs=docs or ["doc1", "doc2", "doc3"],
        scores=scores or [0.9, 0.8, 0.7],
        backend=backend,
        latency_ms=15.0,
    )


@pytest.fixture
def reranker_full():
    return CerebroReranker(mode="full", local_endpoint="http://localhost:8001")


@pytest.fixture
def reranker_shadow():
    return CerebroReranker(mode="shadow", local_endpoint="http://localhost:8001")


@pytest.fixture
def reranker_canary():
    return CerebroReranker(
        mode="canary",
        local_endpoint="http://localhost:8001",
        canary_percentage=50,
    )


class TestShouldUseLocal:
    def test_vertex_only(self):
        r = CerebroReranker(mode="vertex-only")
        assert r._should_use_local() is False
        assert r._should_use_local(user_id="anyone") is False

    def test_full_mode(self, reranker_full):
        assert reranker_full._should_use_local() is True
        assert reranker_full._should_use_local(user_id="anyone") is True

    def test_shadow_mode(self, reranker_shadow):
        # Shadow always serves Vertex
        assert reranker_shadow._should_use_local() is False

    def test_canary_deterministic(self, reranker_canary):
        """Same user_id should always route the same way."""
        result1 = reranker_canary._should_use_local(user_id="user_42")
        result2 = reranker_canary._should_use_local(user_id="user_42")
        assert result1 == result2

    def test_canary_no_user_id(self, reranker_canary):
        """Without user_id, canary defaults to Vertex."""
        assert reranker_canary._should_use_local() is False

    def test_circuit_breaker_forces_vertex(self, reranker_full):
        """When circuit breaker opens, even full mode should route to Vertex."""
        reranker_full.consecutive_errors = 3
        assert reranker_full._should_use_local() is False


class TestCircuitBreaker:
    def test_not_open_initially(self, reranker_full):
        assert reranker_full._circuit_open() is False

    def test_opens_at_threshold(self, reranker_full):
        reranker_full.consecutive_errors = 3
        assert reranker_full._circuit_open() is True

    def test_disabled_when_fallback_off(self):
        r = CerebroReranker(mode="full", fallback_enabled=False)
        r.consecutive_errors = 100
        assert r._circuit_open() is False


class TestAgreement:
    def test_identical_results(self, reranker_full):
        local = _make_result("local", ["a", "b", "c"])
        vertex = _make_result("vertex", ["a", "b", "c"])
        assert reranker_full._compute_agreement(local, vertex) == 1.0

    def test_no_overlap(self, reranker_full):
        local = _make_result("local", ["a", "b", "c"])
        vertex = _make_result("vertex", ["x", "y", "z"])
        assert reranker_full._compute_agreement(local, vertex) == 0.0

    def test_partial_overlap(self, reranker_full):
        local = _make_result("local", ["a", "b", "c", "d"])
        vertex = _make_result("vertex", ["a", "b", "x", "y"])
        agreement = reranker_full._compute_agreement(local, vertex)
        # Jaccard: 2 / 6 = 0.333
        assert 0.3 < agreement < 0.4

    def test_empty_results(self, reranker_full):
        local = _make_result("local", [])
        vertex = _make_result("vertex", [])
        assert reranker_full._compute_agreement(local, vertex) == 0.0


class TestRerankLocal:
    @pytest.mark.asyncio
    async def test_success_resets_errors(self, reranker_full):
        reranker_full.consecutive_errors = 2

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "results": [{"document": "doc1", "score": 0.9}],
            "mode_used": "fast",
            "cache_hit": False,
        }

        reranker_full.local_client.post = AsyncMock(return_value=mock_response)

        result = await reranker_full._rerank_local("query", ["doc1"])
        assert result.backend == "local"
        assert reranker_full.consecutive_errors == 0

    @pytest.mark.asyncio
    async def test_failure_increments_errors(self, reranker_full):
        reranker_full.local_client.post = AsyncMock(
            side_effect=Exception("connection refused")
        )

        with pytest.raises(Exception):
            await reranker_full._rerank_local("query", ["doc1"])

        assert reranker_full.consecutive_errors == 1


class TestRerankShadow:
    @pytest.mark.asyncio
    async def test_shadow_serves_vertex(self, reranker_shadow):
        """Shadow mode should return Vertex result."""
        local_result = _make_result("local")
        vertex_result = _make_result("vertex")

        reranker_shadow._rerank_local = AsyncMock(return_value=local_result)
        reranker_shadow._rerank_vertex = AsyncMock(return_value=vertex_result)

        result = await reranker_shadow.rerank("query", ["doc1", "doc2"])
        assert result.backend == "vertex"

    @pytest.mark.asyncio
    async def test_shadow_local_failure_still_serves_vertex(self, reranker_shadow):
        """If local fails in shadow, Vertex result should still be served."""
        vertex_result = _make_result("vertex")

        reranker_shadow._rerank_local = AsyncMock(
            side_effect=Exception("local down")
        )
        reranker_shadow._rerank_vertex = AsyncMock(return_value=vertex_result)

        result = await reranker_shadow.rerank("query", ["doc1"])
        assert result.backend == "vertex"

    @pytest.mark.asyncio
    async def test_shadow_vertex_failure_serves_local(self, reranker_shadow):
        """If Vertex fails in shadow, local result should be served."""
        local_result = _make_result("local")

        reranker_shadow._rerank_local = AsyncMock(return_value=local_result)
        reranker_shadow._rerank_vertex = AsyncMock(
            side_effect=Exception("vertex down")
        )

        result = await reranker_shadow.rerank("query", ["doc1"])
        assert result.backend == "local"

    @pytest.mark.asyncio
    async def test_shadow_both_fail_raises(self, reranker_shadow):
        """If both backends fail in shadow, should raise."""
        reranker_shadow._rerank_local = AsyncMock(
            side_effect=Exception("local down")
        )
        reranker_shadow._rerank_vertex = AsyncMock(
            side_effect=Exception("vertex down")
        )

        with pytest.raises(Exception, match="vertex down"):
            await reranker_shadow.rerank("query", ["doc1"])


class TestRerankFallback:
    @pytest.mark.asyncio
    async def test_fallback_local_to_vertex(self, reranker_full):
        """Full mode: if local fails and fallback enabled, should try Vertex."""
        vertex_result = _make_result("vertex")

        reranker_full._rerank_local = AsyncMock(
            side_effect=Exception("local error")
        )
        reranker_full._rerank_vertex = AsyncMock(return_value=vertex_result)

        result = await reranker_full.rerank("query", ["doc1"])
        assert result.backend == "vertex"

    @pytest.mark.asyncio
    async def test_no_fallback_raises(self):
        """With fallback disabled, failure should propagate."""
        r = CerebroReranker(mode="full", fallback_enabled=False)
        r._rerank_local = AsyncMock(side_effect=Exception("boom"))

        with pytest.raises(Exception, match="boom"):
            await r.rerank("query", ["doc1"])


class TestRerankResult:
    def test_default_metadata(self):
        r = RerankResult(
            ranked_docs=["a"], scores=[0.9], backend="local", latency_ms=10.0
        )
        assert r.metadata == {}

    def test_custom_metadata(self):
        r = RerankResult(
            ranked_docs=["a"],
            scores=[0.9],
            backend="local",
            latency_ms=10.0,
            metadata={"model": "fast"},
        )
        assert r.metadata["model"] == "fast"
