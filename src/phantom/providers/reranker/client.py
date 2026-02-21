"""
CEREBRO Phantom — Reranker Client

Simplified HTTP client for the reranker service.
Circuit breaker: 3 failures → bypass for 60s.
Fallback: returns documents in original order.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import httpx
import structlog
from prometheus_client import Counter, Histogram

log = structlog.get_logger()

# Metrics
PHANTOM_RERANK_REQUESTS = Counter(
    "phantom_rerank_requests_total", "Phantom rerank requests", ["status"]
)
PHANTOM_RERANK_LATENCY = Histogram(
    "phantom_rerank_duration_seconds", "Phantom rerank latency"
)
PHANTOM_RERANK_ERRORS = Counter(
    "phantom_rerank_errors_total", "Phantom rerank errors", ["error_type"]
)

CIRCUIT_BREAKER_THRESHOLD = 3
CIRCUIT_BREAKER_RESET_SECONDS = 60


@dataclass
class RerankedDocument:
    """A document with its reranker score."""

    content: str
    score: float
    metadata: Dict = field(default_factory=dict)


class PhantomRerankerClient:
    """
    HTTP client for the CEREBRO reranker service.

    Designed for the RAG pipeline: retrieve → rerank → generate.
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:8001",
        timeout: float = 5.0,
        mode: str = "auto",
    ):
        self.endpoint = endpoint
        self.timeout = timeout
        self.mode = mode
        self.client = httpx.AsyncClient(timeout=timeout)

        # Circuit breaker
        self._consecutive_errors = 0
        self._last_failure: float = 0

        log.info(
            "PhantomRerankerClient initialized",
            endpoint=endpoint,
            timeout=timeout,
            mode=mode,
        )

    def _circuit_open(self) -> bool:
        """Check if circuit breaker is tripped."""
        if self._consecutive_errors < CIRCUIT_BREAKER_THRESHOLD:
            return False

        elapsed = time.monotonic() - self._last_failure
        if elapsed >= CIRCUIT_BREAKER_RESET_SECONDS:
            # Half-open: reset and allow one attempt
            self._consecutive_errors = 0
            return False

        return True

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10,
        mode: Optional[str] = None,
    ) -> List[RerankedDocument]:
        """
        Rerank documents via the reranker service.

        Falls back to original order if the service is unavailable.
        """
        if not documents:
            return []

        if self._circuit_open():
            log.warning(
                "Circuit breaker open, returning original order",
                consecutive_errors=self._consecutive_errors,
            )
            PHANTOM_RERANK_REQUESTS.labels(status="circuit_open").inc()
            return self._fallback(documents, top_k)

        start = time.monotonic()

        try:
            response = await self.client.post(
                f"{self.endpoint}/v1/rerank",
                json={
                    "query": query,
                    "documents": documents,
                    "top_k": top_k,
                    "mode": mode or self.mode,
                },
            )
            response.raise_for_status()
            data = response.json()

            latency = time.monotonic() - start
            PHANTOM_RERANK_LATENCY.observe(latency)
            PHANTOM_RERANK_REQUESTS.labels(status="success").inc()

            # Reset circuit breaker on success
            self._consecutive_errors = 0

            return [
                RerankedDocument(
                    content=r["document"],
                    score=r["score"],
                    metadata={
                        "model": r.get("model"),
                        "confidence": r.get("confidence"),
                    },
                )
                for r in data["results"]
            ]

        except Exception as e:
            self._consecutive_errors += 1
            self._last_failure = time.monotonic()
            PHANTOM_RERANK_ERRORS.labels(error_type=type(e).__name__).inc()
            PHANTOM_RERANK_REQUESTS.labels(status="error").inc()

            log.error(
                "Rerank failed, returning original order",
                error=str(e),
                consecutive_errors=self._consecutive_errors,
            )

            return self._fallback(documents, top_k)

    def _fallback(
        self, documents: List[str], top_k: int
    ) -> List[RerankedDocument]:
        """Return documents in original order as fallback."""
        return [
            RerankedDocument(content=doc, score=0.0)
            for doc in documents[:top_k]
        ]

    async def close(self):
        """Cleanup HTTP client."""
        await self.client.aclose()


def create_reranker_client() -> Optional[PhantomRerankerClient]:
    """
    Factory that creates a reranker client if enabled via environment.

    Set CEREBRO_RERANKER_ENABLED=true to enable.
    Set CEREBRO_RERANKER_URL to override the default endpoint.
    """
    if os.getenv("CEREBRO_RERANKER_ENABLED", "").lower() != "true":
        return None

    return PhantomRerankerClient(
        endpoint=os.getenv("CEREBRO_RERANKER_URL", "http://localhost:8001"),
        timeout=float(os.getenv("CEREBRO_RERANKER_TIMEOUT", "5.0")),
        mode=os.getenv("CEREBRO_RERANKER_MODE", "auto"),
    )
