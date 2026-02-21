"""
CEREBRO Reranker Client
Unified interface with migration awareness
"""

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import httpx
import structlog
from prometheus_client import Counter, Histogram

try:
    import xxhash
    def _hash_user(user_id: str) -> int:
        return xxhash.xxh64(user_id.encode()).intdigest()
except ImportError:
    import hashlib
    def _hash_user(user_id: str) -> int:
        return int(hashlib.sha256(user_id.encode()).hexdigest(), 16)

log = structlog.get_logger()

# Metrics
RERANK_REQUESTS = Counter(
    "cerebro_rerank_requests_total", "Rerank requests", ["backend", "mode"]
)
RERANK_LATENCY = Histogram(
    "cerebro_rerank_duration_seconds", "Rerank latency", ["backend"]
)
RERANK_ERRORS = Counter(
    "cerebro_rerank_errors_total", "Rerank errors", ["backend", "error_type"]
)
AGREEMENT_SCORE = Histogram(
    "cerebro_reranker_agreement_score", "Agreement between backends"
)


@dataclass
class RerankResult:
    """Unified rerank result"""

    ranked_docs: List[str]
    scores: List[float]
    backend: str
    latency_ms: float
    metadata: Dict = field(default_factory=dict)


class CerebroReranker:
    """
    Intelligent reranker client with migration support.

    Migration modes:
    - vertex-only: baseline, all traffic to Vertex AI
    - shadow: run both in parallel, serve Vertex, log comparison
    - canary: percentage split via consistent hashing
    - full: 100% local reranking
    """

    def __init__(
        self,
        mode: Literal["vertex-only", "shadow", "canary", "full"] = "shadow",
        local_endpoint: str = "http://localhost:8001",
        fallback_enabled: bool = True,
        canary_percentage: int = 10,
        error_threshold: int = 3,
    ):
        self.mode = mode
        self.local_endpoint = local_endpoint
        self.fallback_enabled = fallback_enabled
        self.canary_percentage = canary_percentage
        self.error_threshold = error_threshold

        # HTTP client
        self.local_client = httpx.AsyncClient(timeout=30.0)

        # Vertex AI client (lazy)
        self._vertex_client = None

        # Circuit breaker state
        self.consecutive_errors = 0

        log.info(
            "CerebroReranker initialized",
            mode=mode,
            fallback=fallback_enabled,
        )

    @property
    def vertex_client(self):
        if self._vertex_client is None and self.mode != "full":
            try:
                from google.cloud import aiplatform

                aiplatform.init()
                self._vertex_client = aiplatform.gapic.PredictionServiceClient()
            except Exception as e:
                log.warning("Vertex AI not configured", error=str(e))
        return self._vertex_client

    def _circuit_open(self) -> bool:
        """Check if circuit breaker is tripped."""
        return (
            self.fallback_enabled
            and self.consecutive_errors >= self.error_threshold
        )

    def _should_use_local(self, user_id: Optional[str] = None) -> bool:
        """Determine which backend to use."""
        if self._circuit_open():
            log.warning(
                "Circuit breaker open, routing to Vertex",
                consecutive_errors=self.consecutive_errors,
            )
            return False

        if self.mode == "vertex-only":
            return False
        elif self.mode == "full":
            return True
        elif self.mode == "canary":
            if user_id:
                bucket = _hash_user(user_id) % 100
                return bucket < self.canary_percentage
            return False
        else:  # shadow — always serve Vertex, local runs in background
            return False

    async def _rerank_local(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10,
        mode: str = "auto",
    ) -> RerankResult:
        """Rerank using local hybrid engine."""
        start = time.monotonic()

        try:
            response = await self.local_client.post(
                f"{self.local_endpoint}/v1/rerank",
                json={
                    "query": query,
                    "documents": documents,
                    "top_k": top_k,
                    "mode": mode,
                },
            )
            response.raise_for_status()
            data = response.json()

            latency_ms = (time.monotonic() - start) * 1000
            self.consecutive_errors = 0

            RERANK_REQUESTS.labels(backend="local", mode=self.mode).inc()
            RERANK_LATENCY.labels(backend="local").observe(latency_ms / 1000)

            return RerankResult(
                ranked_docs=[r["document"] for r in data["results"]],
                scores=[r["score"] for r in data["results"]],
                backend="local",
                latency_ms=latency_ms,
                metadata={
                    "model_used": data.get("mode_used"),
                    "cache_hit": data.get("cache_hit"),
                    "ipfs_cid": data.get("ipfs_cid"),
                },
            )

        except Exception as e:
            self.consecutive_errors += 1
            RERANK_ERRORS.labels(
                backend="local", error_type=type(e).__name__
            ).inc()
            log.error(
                "Local rerank failed",
                error=str(e),
                consecutive_errors=self.consecutive_errors,
            )
            raise

    async def _rerank_vertex(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10,
    ) -> RerankResult:
        """Rerank using Google Vertex AI."""
        if not self.vertex_client:
            raise RuntimeError("Vertex AI not configured")

        start = time.monotonic()

        try:
            # TODO: Implement actual Vertex AI reranking call
            latency_ms = (time.monotonic() - start) * 1000

            RERANK_REQUESTS.labels(backend="vertex", mode=self.mode).inc()
            RERANK_LATENCY.labels(backend="vertex").observe(latency_ms / 1000)

            return RerankResult(
                ranked_docs=documents[:top_k],
                scores=[0.9] * min(top_k, len(documents)),
                backend="vertex",
                latency_ms=latency_ms,
                metadata={"model": "vertex-rerank"},
            )

        except Exception as e:
            RERANK_ERRORS.labels(
                backend="vertex", error_type=type(e).__name__
            ).inc()
            log.error("Vertex rerank failed", error=str(e))
            raise

    def _compute_agreement(
        self,
        local_result: RerankResult,
        vertex_result: RerankResult,
    ) -> float:
        """Jaccard similarity of top-k results."""
        k = 10
        local_set = set(local_result.ranked_docs[:k])
        vertex_set = set(vertex_result.ranked_docs[:k])

        union = len(local_set | vertex_set)
        if union == 0:
            return 0.0

        agreement = len(local_set & vertex_set) / union
        AGREEMENT_SCORE.observe(agreement)
        return agreement

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10,
        user_id: Optional[str] = None,
        mode: str = "auto",
    ) -> RerankResult:
        """
        Main reranking entry point with migration logic.
        """
        # Shadow mode — run both, serve Vertex
        if self.mode == "shadow":
            return await self._rerank_shadow(query, documents, top_k, mode)

        # Canary / full — route based on user
        use_local = self._should_use_local(user_id)

        try:
            if use_local:
                return await self._rerank_local(query, documents, top_k, mode)
            else:
                return await self._rerank_vertex(query, documents, top_k)
        except Exception:
            if not self.fallback_enabled:
                raise

            # Fallback: local failed -> try vertex, vertex failed -> can't recover
            if use_local:
                log.warning("Local failed, falling back to Vertex")
                return await self._rerank_vertex(query, documents, top_k)
            raise

    async def _rerank_shadow(
        self,
        query: str,
        documents: List[str],
        top_k: int,
        mode: str,
    ) -> RerankResult:
        """Shadow mode: run both backends, serve Vertex, log comparison."""
        local_task = asyncio.create_task(
            self._rerank_local(query, documents, top_k, mode)
        )
        vertex_task = asyncio.create_task(
            self._rerank_vertex(query, documents, top_k)
        )

        results = await asyncio.gather(
            local_task, vertex_task, return_exceptions=True
        )
        local_result, vertex_result = results

        # Log comparison if both succeeded
        if not isinstance(local_result, BaseException) and not isinstance(
            vertex_result, BaseException
        ):
            agreement = self._compute_agreement(local_result, vertex_result)
            log.info(
                "Shadow comparison",
                agreement=round(agreement, 3),
                local_latency=round(local_result.latency_ms, 1),
                vertex_latency=round(vertex_result.latency_ms, 1),
            )

        # Serve Vertex if available, otherwise local
        if not isinstance(vertex_result, BaseException):
            return vertex_result
        if not isinstance(local_result, BaseException):
            log.warning("Vertex failed in shadow mode, serving local")
            return local_result

        # Both failed
        raise vertex_result

    async def close(self):
        """Cleanup HTTP client."""
        await self.local_client.aclose()
