"""Tests for CEREBRO RAG engine with reranker integration."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from phantom.core.rag.engine import RAGEngine, RAGResult
from phantom.providers.reranker.client import (
    PhantomRerankerClient,
    RerankedDocument,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_vector_store():
    store = AsyncMock()
    store.search = AsyncMock(
        return_value=[
            "Nginx reverse proxy configuration guide",
            "Apache HTTP server virtual host setup",
            "Docker container networking guide",
            "Nginx load balancing tutorial",
            "SSL certificates with Let's Encrypt",
            "PostgreSQL connection pooling",
            "Redis caching strategies",
            "Kubernetes ingress configuration",
            "HAProxy load balancer setup",
            "Traefik reverse proxy tutorial",
        ]
    )
    return store


@pytest.fixture
def mock_llm_provider():
    provider = AsyncMock()
    provider.grounded_generate = AsyncMock(
        return_value={
            "answer": "To configure nginx as a reverse proxy, use the proxy_pass directive.",
            "metadata": {"model": "test-llm"},
        }
    )
    return provider


@pytest.fixture
def mock_reranker():
    """Reranker client that returns top documents with scores."""
    client = AsyncMock(spec=PhantomRerankerClient)
    client.rerank = AsyncMock(
        return_value=[
            RerankedDocument(
                content="Nginx reverse proxy configuration guide",
                score=0.95,
            ),
            RerankedDocument(
                content="Nginx load balancing tutorial",
                score=0.82,
            ),
            RerankedDocument(
                content="Traefik reverse proxy tutorial",
                score=0.75,
            ),
        ]
    )
    return client


@pytest.fixture
def engine_with_reranker(mock_vector_store, mock_llm_provider, mock_reranker):
    return RAGEngine(
        vector_store=mock_vector_store,
        llm_provider=mock_llm_provider,
        reranker=mock_reranker,
    )


@pytest.fixture
def engine_without_reranker(mock_vector_store, mock_llm_provider):
    return RAGEngine(
        vector_store=mock_vector_store,
        llm_provider=mock_llm_provider,
    )


# ---------------------------------------------------------------------------
# Tests: RAG Engine without reranker
# ---------------------------------------------------------------------------


class TestRAGEngineBasic:
    @pytest.mark.asyncio
    async def test_query_returns_result(self, engine_without_reranker):
        result = await engine_without_reranker.query("how to configure nginx")

        assert isinstance(result, RAGResult)
        assert result.answer != ""
        assert result.reranked is False

    @pytest.mark.asyncio
    async def test_query_uses_top_k(
        self, engine_without_reranker, mock_vector_store
    ):
        await engine_without_reranker.query("query", top_k=3)

        # Without reranker, should fetch exactly top_k
        mock_vector_store.search.assert_called_once_with(
            "query", top_k=3
        )

    @pytest.mark.asyncio
    async def test_query_passes_context_to_llm(
        self, engine_without_reranker, mock_llm_provider
    ):
        await engine_without_reranker.query("query", top_k=3)

        call_kwargs = mock_llm_provider.grounded_generate.call_args.kwargs
        assert len(call_kwargs["context"]) == 3

    @pytest.mark.asyncio
    async def test_query_has_latency(self, engine_without_reranker):
        result = await engine_without_reranker.query("query")
        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_query_no_scores_without_reranker(
        self, engine_without_reranker
    ):
        result = await engine_without_reranker.query("query")
        assert result.scores == []


# ---------------------------------------------------------------------------
# Tests: RAG Engine with reranker
# ---------------------------------------------------------------------------


class TestRAGEngineWithReranker:
    @pytest.mark.asyncio
    async def test_query_uses_reranker(
        self, engine_with_reranker, mock_reranker
    ):
        result = await engine_with_reranker.query("how to configure nginx")

        assert result.reranked is True
        mock_reranker.rerank.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieves_more_candidates(
        self, engine_with_reranker, mock_vector_store
    ):
        """With reranker, should fetch k*3 candidates."""
        await engine_with_reranker.query("query", top_k=5)

        mock_vector_store.search.assert_called_once_with(
            "query", top_k=15  # 5 * 3
        )

    @pytest.mark.asyncio
    async def test_reranker_receives_all_candidates(
        self, engine_with_reranker, mock_reranker
    ):
        await engine_with_reranker.query("query", top_k=5)

        call_args = mock_reranker.rerank.call_args
        # Should receive all 10 documents from vector store
        assert len(call_args.args[1]) == 10
        assert call_args.kwargs["top_k"] == 5

    @pytest.mark.asyncio
    async def test_reranked_results_passed_to_llm(
        self, engine_with_reranker, mock_llm_provider
    ):
        await engine_with_reranker.query("query", top_k=5)

        call_kwargs = mock_llm_provider.grounded_generate.call_args.kwargs
        context = call_kwargs["context"]
        # Should contain reranked docs
        assert "Nginx reverse proxy configuration guide" in context
        assert "Nginx load balancing tutorial" in context

    @pytest.mark.asyncio
    async def test_scores_from_reranker(self, engine_with_reranker):
        result = await engine_with_reranker.query("query")

        assert len(result.scores) == 3
        assert result.scores[0] == 0.95

    @pytest.mark.asyncio
    async def test_use_reranker_false_skips_reranker(
        self, engine_with_reranker, mock_reranker, mock_vector_store
    ):
        """use_reranker=False should bypass reranker even if configured."""
        result = await engine_with_reranker.query(
            "query", top_k=3, use_reranker=False
        )

        assert result.reranked is False
        mock_reranker.rerank.assert_not_called()
        # Should fetch exactly top_k (not k*3)
        mock_vector_store.search.assert_called_once_with(
            "query", top_k=3
        )


# ---------------------------------------------------------------------------
# Tests: Reranker failure / graceful degradation
# ---------------------------------------------------------------------------


class TestRAGEngineRerankerFailure:
    @pytest.mark.asyncio
    async def test_fallback_on_reranker_error(
        self, engine_with_reranker, mock_reranker
    ):
        """If reranker raises, should fall back to original order."""
        mock_reranker.rerank = AsyncMock(
            side_effect=Exception("reranker down")
        )

        result = await engine_with_reranker.query("query", top_k=3)

        assert result.reranked is False
        assert len(result.sources) == 3
        assert result.answer != ""

    @pytest.mark.asyncio
    async def test_fallback_preserves_original_order(
        self, engine_with_reranker, mock_reranker, mock_vector_store
    ):
        """Fallback should return first top_k from vector store."""
        mock_reranker.rerank = AsyncMock(
            side_effect=Exception("timeout")
        )

        result = await engine_with_reranker.query("query", top_k=2)

        # Should be the first 2 documents from vector store
        expected = (await mock_vector_store.search("", top_k=10))[:2]
        assert result.sources == expected


# ---------------------------------------------------------------------------
# Tests: PhantomRerankerClient
# ---------------------------------------------------------------------------


class TestPhantomRerankerClient:
    @pytest.mark.asyncio
    async def test_circuit_breaker_opens(self):
        client = PhantomRerankerClient(endpoint="http://bad:9999", timeout=0.1)
        client.client = AsyncMock()
        client.client.post = AsyncMock(
            side_effect=Exception("connection refused")
        )

        # 3 failures should trip the circuit
        for _ in range(3):
            await client.rerank("q", ["d"])

        assert client._consecutive_errors == 3

    @pytest.mark.asyncio
    async def test_fallback_returns_original_order(self):
        client = PhantomRerankerClient(endpoint="http://bad:9999")
        client.client = AsyncMock()
        client.client.post = AsyncMock(side_effect=Exception("fail"))

        result = await client.rerank("q", ["doc1", "doc2", "doc3"], top_k=2)

        assert len(result) == 2
        assert result[0].content == "doc1"
        assert result[1].content == "doc2"
        assert result[0].score == 0.0  # fallback score

    @pytest.mark.asyncio
    async def test_empty_documents(self):
        client = PhantomRerankerClient()
        result = await client.rerank("q", [])
        assert result == []

    @pytest.mark.asyncio
    async def test_success_resets_errors(self):
        client = PhantomRerankerClient()
        client._consecutive_errors = 2

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"document": "doc1", "score": 0.9, "model": "fast", "confidence": 0.85}
            ]
        }

        client.client = AsyncMock()
        client.client.post = AsyncMock(return_value=mock_response)

        result = await client.rerank("query", ["doc1"])

        assert client._consecutive_errors == 0
        assert len(result) == 1
        assert result[0].content == "doc1"
        assert result[0].score == 0.9
