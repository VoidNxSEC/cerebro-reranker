"""Shared fixtures for CEREBRO Reranker tests."""

import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Model registry fixtures
# ---------------------------------------------------------------------------

SAMPLE_MODELS_TOML = """\
[models.minilm]
name = "ms-marco-MiniLM-L-6-v2"
size_mb = 80
latency_ms = 15
accuracy = 0.89
ipfs_cid = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"

[models.electra]
name = "ms-marco-electra-base"
size_mb = 420
latency_ms = 45
accuracy = 0.93
ipfs_cid = "bafybeie5gq4jxvzmsym6hjlwxej4rwdoxt7wadqvmmwbqi7r27fclha2va"

[models.deberta]
name = "ms-marco-deberta-v3-base"
size_mb = 1400
latency_ms = 120
accuracy = 0.96
ipfs_cid = "bafybeidskjjd4zmr7oh6ku6wp72vvbxyibcli2r6if3ocdcy7jjjusvl2u"

[models.custom]
name = "cerebro-security-reranker"
size_mb = 500
latency_ms = 60
accuracy = 0.97
ipfs_cid = "TBD"
trained_on = "cerebro-knowledge-base"
"""


@pytest.fixture
def models_toml(tmp_path):
    """Create a temporary models.toml file."""
    p = tmp_path / "models.toml"
    p.write_text(SAMPLE_MODELS_TOML)
    return str(p)


@pytest.fixture
def model_registry(models_toml):
    """Create a ModelRegistry with sample data."""
    from models import ModelRegistry

    return ModelRegistry(config_path=models_toml)


# ---------------------------------------------------------------------------
# Cache fixtures
# ---------------------------------------------------------------------------

class FakeRedis:
    """In-memory Redis mock for testing."""

    def __init__(self):
        self._store = {}
        self._ttls = {}

    def ping(self):
        return True

    def get(self, key):
        return self._store.get(key)

    def set(self, key, value):
        self._store[key] = value

    def setex(self, key, ttl, value):
        self._store[key] = value
        self._ttls[key] = ttl

    def delete(self, key):
        self._store.pop(key, None)

    def dbsize(self):
        return len(self._store)

    def flushdb(self):
        self._store.clear()
        self._ttls.clear()


@pytest.fixture
def fake_redis():
    return FakeRedis()


@pytest.fixture
def mock_ipfs_cache(fake_redis):
    """IPFSCache with mocked Redis and IPFS."""
    from cache import IPFSCache

    with patch.object(IPFSCache, "__init__", lambda self, **kw: None):
        c = IPFSCache.__new__(IPFSCache)
        c.redis = fake_redis
        c.ipfs = None  # no IPFS in tests
        c.ttl = 3600
        c.redis_url = "redis://localhost:6379"
        c.ipfs_api = "/ip4/127.0.0.1/tcp/5001"
        return c


# ---------------------------------------------------------------------------
# Reranker fixtures
# ---------------------------------------------------------------------------

def _make_fake_cross_encoder():
    """CrossEncoder mock that returns deterministic scores."""
    mock = MagicMock()

    def predict(pairs, batch_size=32, show_progress_bar=False):
        # Score based on simple word overlap
        scores = []
        for query, doc in pairs:
            q_words = set(query.lower().split())
            d_words = set(doc.lower().split())
            overlap = len(q_words & d_words) / max(len(q_words | d_words), 1)
            scores.append(overlap)
        return np.array(scores)

    mock.predict = predict
    return mock


@pytest.fixture
def hybrid_reranker(model_registry, mock_ipfs_cache):
    """HybridReranker with mocked models."""
    from hybrid_engine import HybridReranker

    with patch.object(HybridReranker, "_load_models"):
        reranker = HybridReranker(
            model_registry=model_registry,
            cache=mock_ipfs_cache,
            confidence_threshold=0.8,
            max_batch_size=32,
            device="cpu",
        )
        reranker.models = {
            "fast": _make_fake_cross_encoder(),
            "accurate": _make_fake_cross_encoder(),
        }
        return reranker


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_QUERY = "how to configure nginx reverse proxy"

SAMPLE_DOCUMENTS = [
    "Nginx reverse proxy configuration guide with upstream directives",
    "Apache HTTP server virtual host setup tutorial",
    "Docker container networking and port forwarding",
    "Nginx load balancing across multiple backends",
    "Setting up SSL certificates with Let's Encrypt",
    "PostgreSQL connection pooling with PgBouncer",
    "Redis caching strategies for web applications",
    "Kubernetes ingress controller configuration",
]


@pytest.fixture
def sample_query():
    return SAMPLE_QUERY


@pytest.fixture
def sample_documents():
    return SAMPLE_DOCUMENTS
