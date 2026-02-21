"""
CEREBRO Phantom — RAG Engine

Pipeline: retrieve → rerank (optional) → generate
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

import structlog
from prometheus_client import Counter, Histogram

log = structlog.get_logger()

# Metrics
RAG_QUERIES = Counter("phantom_rag_queries_total", "RAG queries", ["reranked"])
RAG_LATENCY = Histogram(
    "phantom_rag_query_duration_seconds", "RAG query latency", ["stage"]
)


class VectorStoreProvider(Protocol):
    """Interface for vector store backends."""

    async def search(self, query: str, top_k: int) -> List[str]: ...


class LLMProvider(Protocol):
    """Interface for LLM generation backends."""

    async def grounded_generate(
        self, query: str, context: List[str], top_k: int
    ) -> Dict[str, Any]: ...


@dataclass
class RAGResult:
    """Result from a RAG query."""

    answer: str
    sources: List[str]
    scores: List[float]
    reranked: bool
    latency_ms: float
    metadata: Dict = field(default_factory=dict)


class RAGEngine:
    """
    RAG engine with optional reranker integration.

    Without reranker:
        retrieve(k) → generate

    With reranker:
        retrieve(k*3) → rerank(top_k=k) → generate
    """

    def __init__(
        self,
        vector_store: VectorStoreProvider,
        llm_provider: LLMProvider,
        reranker: Optional[Any] = None,
    ):
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        self.reranker = reranker

        log.info(
            "RAGEngine initialized",
            reranker_enabled=reranker is not None,
        )

    async def query(
        self,
        query: str,
        top_k: int = 5,
        use_reranker: bool = True,
    ) -> RAGResult:
        """
        Execute a RAG query with optional reranking.

        Args:
            query: The user's question.
            top_k: Number of documents to use as context.
            use_reranker: Whether to use reranker (if available).
        """
        return await self.query_with_metrics(query, top_k, use_reranker)

    async def query_with_metrics(
        self,
        query: str,
        top_k: int = 5,
        use_reranker: bool = True,
    ) -> RAGResult:
        """RAG query with Prometheus instrumentation."""
        start = time.monotonic()
        reranked = False

        # Step 1: Retrieve candidates
        # Fetch more candidates when reranker is available to give it margin
        retrieve_k = top_k * 3 if (self.reranker and use_reranker) else top_k

        with RAG_LATENCY.labels(stage="retrieve").time():
            candidates = await self.vector_store.search(query, top_k=retrieve_k)

        # Step 2: Rerank if available and requested
        scores: List[float] = []
        if self.reranker and use_reranker:
            try:
                with RAG_LATENCY.labels(stage="rerank").time():
                    reranked_docs = await self.reranker.rerank(
                        query, candidates, top_k=top_k
                    )

                # Extract content and scores from reranked results
                candidates = [doc.content for doc in reranked_docs]
                scores = [doc.score for doc in reranked_docs]
                reranked = True

            except Exception as e:
                log.warning(
                    "Reranker failed, using original order",
                    error=str(e),
                )
                candidates = candidates[:top_k]
        else:
            candidates = candidates[:top_k]

        # Step 3: Generate with refined context
        with RAG_LATENCY.labels(stage="generate").time():
            result = await self.llm_provider.grounded_generate(
                query=query, context=candidates, top_k=top_k
            )

        latency_ms = (time.monotonic() - start) * 1000

        RAG_QUERIES.labels(reranked=str(reranked).lower()).inc()

        return RAGResult(
            answer=result.get("answer", ""),
            sources=candidates,
            scores=scores,
            reranked=reranked,
            latency_ms=latency_ms,
            metadata=result.get("metadata", {}),
        )
