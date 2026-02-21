"""
IPFS-backed Redis Cache
Persistent caching with distributed storage
"""

import hashlib
import json
from typing import Optional, Dict, Any

import redis.asyncio as aioredis
import ipfshttpclient
import structlog

log = structlog.get_logger()


class IPFSCache:
    """
    Two-tier cache:
    1. Redis for hot data (fast, async-safe)
    2. IPFS for cold data (persistent, distributed)
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        ipfs_api: str = "/ip4/127.0.0.1/tcp/5001",
        ttl: int = 3600
    ):
        self.redis_url = redis_url
        self.ipfs_api = ipfs_api
        self.ttl = ttl
        self._redis: Optional[aioredis.Redis] = None

        # Connect to IPFS (sync client — IPFS operations are fine blocking)
        try:
            self.ipfs = ipfshttpclient.connect(ipfs_api)
            log.info("IPFS connected", api=ipfs_api)
        except Exception as e:
            log.warning("IPFS connection failed", error=str(e))
            self.ipfs = None

    async def _get_redis(self) -> Optional[aioredis.Redis]:
        """Lazily create and verify the async Redis connection."""
        if self._redis is None:
            try:
                self._redis = aioredis.from_url(
                    self.redis_url, decode_responses=True
                )
                await self._redis.ping()
                log.info("Redis connected", url=self.redis_url)
            except Exception as e:
                log.error("Redis connection failed", error=str(e))
                self._redis = None
        return self._redis

    def is_connected(self) -> bool:
        """Non-blocking check: True if the async client was successfully created."""
        return self._redis is not None

    async def get(self, key: str) -> Optional[Dict]:
        """Get cached result (async-safe)."""
        redis = await self._get_redis()
        if not redis:
            return None

        try:
            # Hot cache: Redis
            cached = await redis.get(f"rerank:{key}")
            if cached:
                log.debug("Cache hit (Redis)", key=key[:16])
                return json.loads(cached)

            # Cold cache: IPFS
            ipfs_cid = await redis.get(f"rerank:ipfs:{key}")
            if ipfs_cid and self.ipfs:
                try:
                    data = self.ipfs.cat(ipfs_cid)
                    result = json.loads(data)

                    # Warm up Redis
                    await redis.setex(f"rerank:{key}", self.ttl, json.dumps(result))

                    log.debug("Cache hit (IPFS)", key=key[:16], cid=ipfs_cid)
                    return result

                except Exception as e:
                    log.warning("IPFS fetch failed", error=str(e))

            return None

        except Exception as e:
            log.error("Cache get failed", error=str(e))
            return None

    async def set(self, key: str, value: Dict) -> Optional[str]:
        """Set cached result and return IPFS CID (async-safe)."""
        redis = await self._get_redis()
        if not redis:
            return None

        try:
            serialized = json.dumps(value)

            # Hot cache: Redis
            await redis.setex(f"rerank:{key}", self.ttl, serialized)

            # Cold cache: IPFS
            ipfs_cid = None
            if self.ipfs:
                try:
                    ipfs_cid = self.ipfs.add_json(value)

                    # Store CID reference with extended TTL
                    await redis.setex(f"rerank:ipfs:{key}", self.ttl * 24, ipfs_cid)

                    # Pin permanently
                    self.ipfs.pin.add(ipfs_cid)

                    log.debug("Cached to IPFS", key=key[:16], cid=ipfs_cid)

                except Exception as e:
                    log.warning("IPFS pin failed", error=str(e))

            return ipfs_cid

        except Exception as e:
            log.error("Cache set failed", error=str(e))
            return None

    async def size(self) -> int:
        """Get number of Redis keys (async-safe)."""
        redis = await self._get_redis()
        if not redis:
            return 0
        try:
            return await redis.dbsize()
        except Exception:
            return 0

    async def clear(self) -> None:
        """Clear all Redis keys (async-safe)."""
        redis = await self._get_redis()
        if redis:
            await redis.flushdb()
            log.info("Cache cleared")
