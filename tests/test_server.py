"""Tests for the FastAPI server."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def client(hybrid_reranker, mock_ipfs_cache, model_registry):
    """Create a TestClient with mocked dependencies."""
    import server

    server.reranker = hybrid_reranker
    server.cache = mock_ipfs_cache
    server.model_registry = model_registry

    return TestClient(server.app, raise_server_exceptions=False)


class TestRerankEndpoint:
    def test_rerank_basic(self, client):
        resp = client.post(
            "/v1/rerank",
            json={
                "query": "nginx proxy setup",
                "documents": ["nginx config guide", "redis caching tutorial"],
                "top_k": 2,
                "mode": "fast",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 2
        assert "mode_used" in data
        assert "cache_hit" in data
        assert "latency_ms" in data

    def test_rerank_default_params(self, client):
        resp = client.post(
            "/v1/rerank",
            json={
                "query": "test query",
                "documents": ["doc1", "doc2", "doc3"],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) <= 10  # default top_k

    def test_rerank_all_modes(self, client):
        for mode in ["auto", "fast", "accurate"]:
            resp = client.post(
                "/v1/rerank",
                json={
                    "query": "test",
                    "documents": ["doc1", "doc2"],
                    "mode": mode,
                },
            )
            assert resp.status_code == 200, f"mode={mode} failed"

    def test_rerank_empty_documents(self, client):
        resp = client.post(
            "/v1/rerank",
            json={
                "query": "test",
                "documents": [],
            },
        )
        assert resp.status_code == 422  # validation error

    def test_rerank_missing_query(self, client):
        resp = client.post(
            "/v1/rerank",
            json={
                "documents": ["doc1"],
            },
        )
        assert resp.status_code == 422

    def test_rerank_result_structure(self, client):
        resp = client.post(
            "/v1/rerank",
            json={
                "query": "test",
                "documents": ["doc1"],
                "top_k": 1,
                "mode": "fast",
            },
        )
        assert resp.status_code == 200
        result = resp.json()["results"][0]
        assert "document" in result
        assert "score" in result
        assert "model" in result
        assert "confidence" in result
        assert 0.0 <= result["score"] <= 1.0
        assert 0.0 <= result["confidence"] <= 1.0

    def test_rerank_scores_sorted(self, client):
        resp = client.post(
            "/v1/rerank",
            json={
                "query": "test query",
                "documents": [f"document {i}" for i in range(10)],
                "top_k": 10,
                "mode": "fast",
            },
        )
        assert resp.status_code == 200
        scores = [r["score"] for r in resp.json()["results"]]
        assert scores == sorted(scores, reverse=True)


class TestHealthEndpoint:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "models_loaded" in data
        assert isinstance(data["models_loaded"], list)
        assert "cache_size" in data

    def test_health_models_listed(self, client):
        resp = client.get("/health")
        models = resp.json()["models_loaded"]
        assert "minilm" in models
        assert "deberta" in models


class TestMetricsEndpoint:
    def test_metrics_returns_data(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200


class TestModelsEndpoint:
    def test_list_models(self, client):
        resp = client.get("/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert "minilm" in data["models"]
