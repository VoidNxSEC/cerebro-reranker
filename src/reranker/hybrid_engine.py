"""
Hybrid Reranking Engine
Adaptive scoring with fallback strategies
"""

import asyncio
import hashlib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import torch
from sentence_transformers import CrossEncoder
import structlog
from contextlib import asynccontextmanager

from models import ModelRegistry
from cache import IPFSCache

log = structlog.get_logger()


@dataclass
class RerankResult:
    """Single reranking result"""
    document: str
    score: float
    model: str
    confidence: float
    cached: bool = False


class AdaptiveBatcher:
    """Dynamic batch size optimization based on GPU memory"""

    def __init__(self, initial_size: int = 32, min_size: int = 4, max_size: int = 64):
        self.current_size = initial_size
        self.min_size = min_size
        self.max_size = max_size
        self.oom_count = 0
        self.success_count = 0

    def adjust(self, success: bool):
        """Adjust batch size based on OOM events"""
        if not success:
            self.oom_count += 1
            self.current_size = max(self.min_size, self.current_size // 2)
            log.warning("OOM detected, reducing batch size", new_size=self.current_size)
        else:
            self.success_count += 1
            # Gradually increase after 10 successful batches
            if self.success_count >= 10:
                self.current_size = min(self.max_size, int(self.current_size * 1.2))
                self.success_count = 0
                log.info("Increasing batch size", new_size=self.current_size)

        return self.current_size

    @property
    def size(self) -> int:
        return self.current_size


class CircuitBreaker:
    """Circuit breaker for cloud fallback"""

    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.last_failure_time = 0
        self.state = 'closed'  # closed, open, half_open

    def record_success(self):
        """Reset on success"""
        self.failure_count = 0
        if self.state == 'half_open':
            self.state = 'closed'
            log.info("Circuit breaker closed")

    def record_failure(self):
        """Track failures"""
        self.failure_count += 1
        self.last_failure_time = asyncio.get_event_loop().time()

        if self.failure_count >= self.failure_threshold:
            self.state = 'open'
            log.warning(
                "Circuit breaker opened",
                failures=self.failure_count,
                timeout=self.timeout
            )

    def can_attempt(self) -> bool:
        """Check if we can attempt cloud call"""
        if self.state == 'closed':
            return True

        if self.state == 'open':
            # Check if timeout passed
            current_time = asyncio.get_event_loop().time()
            if current_time - self.last_failure_time > self.timeout:
                self.state = 'half_open'
                log.info("Circuit breaker half-open, attempting recovery")
                return True
            return False

        # half_open - allow one attempt
        return True


class HybridReranker:
    """
    Hybrid reranking engine with multiple strategies:
    - Fast local model for high-confidence cases
    - Accurate local model for uncertainty
    - Cloud fallback for critical queries
    """

    def __init__(
        self,
        model_registry: ModelRegistry,
        cache: IPFSCache,
        confidence_threshold: float = 0.8,
        max_batch_size: int = 32,
        device: Optional[str] = None
    ):
        self.model_registry = model_registry
        self.cache = cache
        self.confidence_threshold = confidence_threshold

        # Device selection
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        log.info("Initializing HybridReranker", device=self.device)

        # Load models
        self.models = {}
        self._load_models()

        # Adaptive batching
        self.batcher = AdaptiveBatcher(initial_size=max_batch_size)

        # Circuit breaker for cloud
        self.cloud_breaker = CircuitBreaker()

        # Stats
        self.stats = {
            'requests': 0,
            'cache_hits': 0,
            'fast_model': 0,
            'accurate_model': 0,
            'cloud_fallback': 0,
            'errors': 0
        }

    def _load_models(self):
        """Load all registered models"""
        for model_id in ['fast', 'accurate']:
            try:
                model_info = self.model_registry.get_model(model_id)

                # Check IPFS first
                if model_info.get('ipfs_cid'):
                    model_path = self._fetch_from_ipfs(model_info['ipfs_cid'])
                else:
                    model_path = model_info['name']

                self.models[model_id] = CrossEncoder(
                    model_path,
                    max_length=512,
                    device=self.device
                )

                log.info(
                    "Model loaded",
                    model_id=model_id,
                    device=self.device,
                    memory_mb=self._get_model_memory(model_id)
                )

            except Exception as e:
                log.error(f"Failed to load {model_id} model", error=str(e))

    def _fetch_from_ipfs(self, cid: str) -> str:
        """Fetch model from IPFS"""
        import ipfshttpclient

        client = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001')

        # Download to cache
        cache_path = f"/var/cache/cerebro-reranker/models/{cid}"
        client.get(cid, target=cache_path)

        return cache_path

    def _get_model_memory(self, model_id: str) -> float:
        """Get model memory usage in MB"""
        if self.device == 'cuda':
            torch.cuda.synchronize()
            return torch.cuda.memory_allocated() / 1024**2
        return 0

    def _generate_cache_key(self, query: str, documents: List[str]) -> str:
        """Generate deterministic cache key"""
        content = f"{query}|{'|'.join(sorted(documents))}"
        return hashlib.sha256(content.encode()).hexdigest()

    async def _score_with_model(
        self,
        model_id: str,
        query: str,
        documents: List[str]
    ) -> Tuple[np.ndarray, bool]:
        """Score documents with a specific model"""
        pairs = [[query, doc] for doc in documents]

        try:
            # Adaptive batching
            batch_size = self.batcher.size

            with torch.cuda.amp.autocast(enabled=(self.device == 'cuda')):
                scores = self.models[model_id].predict(
                    pairs,
                    batch_size=batch_size,
                    show_progress_bar=False
                )

            self.batcher.adjust(success=True)
            return scores, True

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                # OOM - retry with smaller batch
                torch.cuda.empty_cache()
                self.batcher.adjust(success=False)

                log.warning(
                    "OOM during inference, retrying",
                    model=model_id,
                    new_batch_size=self.batcher.size
                )

                # Retry
                scores = self.models[model_id].predict(
                    pairs,
                    batch_size=self.batcher.size,
                    show_progress_bar=False
                )
                return scores, True
            else:
                raise

    def _compute_confidence(self, scores: np.ndarray) -> float:
        """
        Compute confidence metric:
        - High variance in scores = high confidence in top results
        - Low variance = uncertain, need better model
        """
        if len(scores) < 2:
            return 1.0

        # Normalize scores
        norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

        # Confidence = variance + max score
        variance = np.var(norm_scores)
        max_score = norm_scores.max()

        # Weighted combination
        confidence = 0.7 * max_score + 0.3 * variance

        return float(np.clip(confidence, 0, 1))

    async def _cloud_rerank(
        self,
        query: str,
        documents: List[str]
    ) -> Optional[np.ndarray]:
        """Fallback to GCP Vertex AI"""

        if not self.cloud_breaker.can_attempt():
            log.warning("Circuit breaker open, skipping cloud")
            return None

        try:
            from google.cloud import aiplatform

            # TODO: Implement Vertex AI reranking
            # This is a placeholder - you'll need to configure your endpoint

            log.info("Cloud reranking not implemented yet")
            return None

        except Exception as e:
            log.error("Cloud rerank failed", error=str(e))
            self.cloud_breaker.record_failure()
            return None

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10,
        mode: str = 'auto',
        use_cache: bool = True
    ) -> Dict:
        """
        Main reranking logic with hybrid strategy
        """
        self.stats['requests'] += 1

        # Check cache
        cache_key = self._generate_cache_key(query, documents)

        if use_cache:
            cached = await self.cache.get(cache_key)
            if cached:
                self.stats['cache_hits'] += 1
                return {
                    'results': cached['results'],
                    'mode_used': cached.get('mode_used', 'cached'),
                    'model_used': cached.get('model_used', 'cached'),
                    'cache_hit': True,
                    'ipfs_cid': cached.get('ipfs_cid')
                }

        # Rerank based on mode
        if mode == 'fast':
            scores, success = await self._score_with_model('fast', query, documents)
            model_used = 'fast'
            confidence = self._compute_confidence(scores)
            self.stats['fast_model'] += 1

        elif mode == 'accurate':
            scores, success = await self._score_with_model('accurate', query, documents)
            model_used = 'accurate'
            confidence = self._compute_confidence(scores)
            self.stats['accurate_model'] += 1

        elif mode == 'cloud':
            scores = await self._cloud_rerank(query, documents)
            if scores is None:
                # Fallback to accurate
                scores, success = await self._score_with_model('accurate', query, documents)
                model_used = 'accurate_fallback'
            else:
                model_used = 'cloud'
            confidence = 1.0
            self.stats['cloud_fallback'] += 1

        else:  # auto
            # Step 1: Fast model
            scores, success = await self._score_with_model('fast', query, documents)
            confidence = self._compute_confidence(scores)

            # Step 2: Check confidence
            if confidence >= self.confidence_threshold:
                # High confidence - use fast result
                model_used = 'fast'
                self.stats['fast_model'] += 1
                log.debug(
                    "Fast model sufficient",
                    confidence=confidence,
                    threshold=self.confidence_threshold
                )
            else:
                # Low confidence - use accurate model
                log.debug(
                    "Low confidence, using accurate model",
                    confidence=confidence,
                    threshold=self.confidence_threshold
                )
                scores, success = await self._score_with_model('accurate', query, documents)
                confidence = self._compute_confidence(scores)
                model_used = 'accurate'
                self.stats['accurate_model'] += 1

        # Sort and truncate
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = [
            {
                'document': documents[idx],
                'score': float(scores[idx]),
                'model': model_used,
                'confidence': confidence
            }
            for idx in ranked_indices
        ]

        # Cache result
        result_dict = {
            'results': results,
            'mode_used': mode,
            'model_used': model_used,
            'cache_hit': False
        }

        if use_cache:
            ipfs_cid = await self.cache.set(cache_key, result_dict)
            result_dict['ipfs_cid'] = ipfs_cid

        return result_dict

    def get_stats(self) -> Dict:
        """Get runtime statistics"""
        return {
            **self.stats,
            'cache_hit_rate': self.stats['cache_hits'] / max(self.stats['requests'], 1),
            'current_batch_size': self.batcher.size,
            'circuit_breaker_state': self.cloud_breaker.state,
            'gpu_memory_mb': self._get_model_memory('fast') if 'fast' in self.models else 0
        }
