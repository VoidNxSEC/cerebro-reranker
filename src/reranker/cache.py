"""
IPFS-backed Redis Cache
Persistent caching with distributed storage
"""

import json
import redis
import ipfshttpclient
from typing import Optional, Dict, Any
import structlog
from datetime import timedelta

log = structlog.get_logger()


class IPFSCache:
    """
    Two-tier cache:
    1. Redis for hot data (fast)
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

        # Connect to Redis
        try:
            self.redis = redis.from_url(redis_url, decode_responses=True)
            self.redis.ping()
            log.info("Redis connected", url=redis_url)
        except Exception as e:
            log.error("Redis connection failed", error=str(e))
            self.redis = None

        # Connect to IPFS
        try:
            self.ipfs = ipfshttpclient.connect(ipfs_api)
            log.info("IPFS connected", api=ipfs_api)
        except Exception as e:
            log.warning("IPFS connection failed", error=str(e))
            self.ipfs = None

    def is_connected(self) -> bool:
        """Check if both Redis and IPFS are connected"""
        redis_ok = self.redis is not None and self.redis.ping()
        ipfs_ok = self.ipfs is not None
        return redis_ok and ipfs_ok

    async def get(self, key: str) -> Optional[Dict]:
        """Get cached result"""
        if not self.redis:
            return None

        try:
            # Check Redis first (hot cache)
            cached = self.redis.get(f"rerank:{key}")

            if cached:
                log.debug("Cache hit (Redis)", key=key[:16])
                return json.loads(cached)

            # Check IPFS (cold cache)
            ipfs_key = f"rerank:ipfs:{key}"
            ipfs_cid = self.redis.get(ipfs_key)

            if ipfs_cid and self.ipfs:
                try:
                    data = self.ipfs.cat(ipfs_cid)
                    result = json.loads(data)

                    # Warm up Redis
                    self.redis.setex(
                        f"rerank:{key}",
                        self.ttl,
                        json.dumps(result)
                    )

                    log.debug("Cache hit (IPFS)", key=key[:16], cid=ipfs_cid)
                    return result

                except Exception as e:
                    log.warning("IPFS fetch failed", error=str(e))

            return None

        except Exception as e:
            log.error("Cache get failed", error=str(e))
            return None

    async def set(self, key: str, value: Dict) -> Optional[str]:
        """Set cached result and return IPFS CID"""
        if not self.redis:
            return None

        try:
            serialized = json.dumps(value)

            # Set in Redis (hot cache)
            self.redis.setex(
                f"rerank:{key}",
                self.ttl,
                serialized
            )

            # Pin to IPFS (cold cache)
            ipfs_cid = None
            if self.ipfs:
                try:
                    result = self.ipfs.add_json(value)
                    ipfs_cid = result

                    # Store CID reference
                    self.redis.setex(
                        f"rerank:ipfs:{key}",
                        self.ttl * 24,  # Keep CID longer
                        ipfs_cid
                    )

                    # Pin permanently
                    self.ipfs.pin.add(ipfs_cid)

                    log.debug(
                        "Cached to IPFS",
                        key=key[:16],
                        cid=ipfs_cid
                    )

                except Exception as e:
                    log.warning("IPFS pin failed", error=str(e))

            return ipfs_cid

        except Exception as e:
            log.error("Cache set failed", error=str(e))
            return None

    def size(self) -> int:
        """Get cache size (number of keys)"""
        if not self.redis:
            return 0

        try:
            return self.redis.dbsize()
        except:
            return 0

    def clear(self):
        """Clear all cache"""
        if self.redis:
            self.redis.flushdb()
            log.info("Cache cleared")
