#!/usr/bin/env python3
"""
CEREBRO Hybrid Reranker API Server
Production-ready FastAPI service with IPFS-backed caching
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from typing import List, Optional, Literal
import structlog
import os
import time

from hybrid_engine import HybridReranker
from cache import IPFSCache
from models import ModelRegistry

# Structured logging
log = structlog.get_logger()

# Metrics
REQUESTS = Counter('rerank_requests_total', 'Total rerank requests', ['model', 'mode'])
ERRORS = Counter('rerank_errors_total', 'Total errors', ['error_type'])
LATENCY = Histogram('rerank_duration_seconds', 'Request latency', ['model'])
CACHE_HITS = Counter('rerank_cache_hits_total', 'Cache hits')
CACHE_MISSES = Counter('rerank_cache_misses_total', 'Cache misses')
CONFIDENCE = Histogram('rerank_confidence_score', 'Confidence score distribution')
GPU_MEMORY = Gauge('rerank_gpu_memory_bytes', 'GPU memory usage')

# App
app = FastAPI(
    title="CEREBRO Hybrid Reranker",
    description="Enterprise semantic search reranking with IPFS distribution",
    version="1.0.0"
)

# Global state
reranker: Optional[HybridReranker] = None
cache: Optional[IPFSCache] = None
model_registry: Optional[ModelRegistry] = None


# Request/Response models
class RerankRequest(BaseModel):
    query: str = Field(..., description="Search query")
    documents: List[str] = Field(..., min_items=1, max_items=1000, description="Documents to rerank")
    top_k: int = Field(10, ge=1, le=100, description="Number of results to return")
    mode: Literal['auto', 'fast', 'accurate', 'cloud'] = Field('auto', description="Reranking mode")
    use_cache: bool = Field(True, description="Enable caching")


class RerankResult(BaseModel):
    document: str
    score: float = Field(..., ge=0.0, le=1.0)
    model: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class RerankResponse(BaseModel):
    results: List[RerankResult]
    mode_used: str
    cache_hit: bool
    latency_ms: float
    ipfs_cid: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    ipfs_connected: bool
    cache_size: int


@app.on_event("startup")
async def startup():
    """Initialize services"""
    global reranker, cache, model_registry

    log.info("Starting CEREBRO Reranker...")

    try:
        # Model registry
        model_registry = ModelRegistry(
            config_path=os.getenv('MODEL_REGISTRY', '/app/models.toml')
        )

        # Cache layer
        cache = IPFSCache(
            redis_url=os.getenv('REDIS_URL', 'redis://localhost:6379'),
            ipfs_api=os.getenv('IPFS_API', '/ip4/127.0.0.1/tcp/5001'),
            ttl=int(os.getenv('CACHE_TTL', '3600'))
        )

        # Reranker engine
        reranker = HybridReranker(
            model_registry=model_registry,
            cache=cache,
            confidence_threshold=float(os.getenv('CONFIDENCE_THRESHOLD', '0.8')),
            max_batch_size=int(os.getenv('MAX_BATCH_SIZE', '32'))
        )

        log.info(
            "Startup complete",
            models=model_registry.list_models(),
            cache_connected=cache.is_connected()
        )

    except Exception as e:
        log.error("Startup failed", error=str(e))
        raise


@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank(req: RerankRequest):
    """Rerank documents for a query"""
    start_time = time.time()

    try:
        # Metrics
        REQUESTS.labels(model='unknown', mode=req.mode).inc()

        # Rerank
        result = await reranker.rerank(
            query=req.query,
            documents=req.documents,
            top_k=req.top_k,
            mode=req.mode,
            use_cache=req.use_cache
        )

        # Update metrics
        latency = (time.time() - start_time) * 1000
        LATENCY.labels(model=result['model_used']).observe(latency / 1000)

        if result['cache_hit']:
            CACHE_HITS.inc()
        else:
            CACHE_MISSES.inc()

        for r in result['results']:
            CONFIDENCE.observe(r['confidence'])

        # Response
        return RerankResponse(
            results=[
                RerankResult(
                    document=r['document'],
                    score=r['score'],
                    model=r['model'],
                    confidence=r['confidence']
                )
                for r in result['results']
            ],
            mode_used=result['mode_used'],
            cache_hit=result['cache_hit'],
            latency_ms=latency,
            ipfs_cid=result.get('ipfs_cid')
        )

    except Exception as e:
        ERRORS.labels(error_type=type(e).__name__).inc()
        log.error("Rerank failed", error=str(e), query=req.query[:100])
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check"""
    return HealthResponse(
        status="ok" if reranker else "not_ready",
        models_loaded=model_registry.list_models() if model_registry else [],
        ipfs_connected=cache.is_connected() if cache else False,
        cache_size=cache.size() if cache else 0
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics"""
    return generate_latest()


@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "models": model_registry.get_all_models() if model_registry else []
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global error handler"""
    ERRORS.labels(error_type="unhandled").inc()
    log.error("Unhandled exception", error=str(exc), path=request.url.path)

    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv('PORT', '8000')),
        log_level="info",
        access_log=True
    )
