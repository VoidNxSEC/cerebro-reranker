"""Tests for IPFSCache."""

import json
import pytest
from unittest.mock import MagicMock, patch
from cache import IPFSCache


class TestIPFSCache:
    @pytest.mark.asyncio
    async def test_get_miss(self, mock_ipfs_cache):
        result = await mock_ipfs_cache.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_and_get(self, mock_ipfs_cache):
        data = {"results": [{"document": "doc1", "score": 0.9}]}
        await mock_ipfs_cache.set("key1", data)

        result = await mock_ipfs_cache.get("key1")
        assert result is not None
        assert result["results"][0]["document"] == "doc1"
        assert result["results"][0]["score"] == 0.9

    @pytest.mark.asyncio
    async def test_set_returns_none_without_ipfs(self, mock_ipfs_cache):
        """Without IPFS, set should still work (Redis only) but return None CID."""
        cid = await mock_ipfs_cache.set("key1", {"data": "value"})
        assert cid is None  # no IPFS configured

    @pytest.mark.asyncio
    async def test_set_with_ipfs(self, mock_ipfs_cache):
        """With IPFS, set should return a CID."""
        mock_ipfs = MagicMock()
        mock_ipfs.add_json.return_value = "bafytest123"
        mock_ipfs.pin = MagicMock()
        mock_ipfs_cache.ipfs = mock_ipfs

        cid = await mock_ipfs_cache.set("key1", {"data": "value"})
        assert cid == "bafytest123"
        mock_ipfs.pin.add.assert_called_once_with("bafytest123")

    @pytest.mark.asyncio
    async def test_get_from_ipfs_cold_cache(self, mock_ipfs_cache, fake_redis):
        """When Redis hot cache misses but IPFS CID exists, fetch from IPFS."""
        cold_data = {"results": [{"document": "cold_doc", "score": 0.7}]}

        mock_ipfs = MagicMock()
        mock_ipfs.cat.return_value = json.dumps(cold_data).encode()
        mock_ipfs_cache.ipfs = mock_ipfs

        # Simulate: no hot cache, but CID reference exists
        fake_redis.set("rerank:ipfs:cold_key", "bafycoldcid")

        result = await mock_ipfs_cache.get("cold_key")
        assert result is not None
        assert result["results"][0]["document"] == "cold_doc"

        # Should have warmed up Redis
        hot = fake_redis.get("rerank:cold_key")
        assert hot is not None

    def test_is_connected_both(self, mock_ipfs_cache):
        mock_ipfs_cache.ipfs = MagicMock()
        assert mock_ipfs_cache.is_connected() is True

    def test_is_connected_no_redis(self, mock_ipfs_cache):
        mock_ipfs_cache.redis = None
        assert mock_ipfs_cache.is_connected() is False

    def test_size(self, mock_ipfs_cache, fake_redis):
        fake_redis.set("a", "1")
        fake_redis.set("b", "2")
        assert mock_ipfs_cache.size() == 2

    def test_size_no_redis(self, mock_ipfs_cache):
        mock_ipfs_cache.redis = None
        assert mock_ipfs_cache.size() == 0

    def test_clear(self, mock_ipfs_cache, fake_redis):
        fake_redis.set("a", "1")
        fake_redis.set("b", "2")
        mock_ipfs_cache.clear()
        assert fake_redis.dbsize() == 0

    @pytest.mark.asyncio
    async def test_get_redis_error(self, mock_ipfs_cache, fake_redis):
        """Cache get should return None on Redis errors, not raise."""
        fake_redis.get = MagicMock(side_effect=Exception("connection lost"))
        result = await mock_ipfs_cache.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_redis_error(self, mock_ipfs_cache, fake_redis):
        """Cache set should return None on Redis errors, not raise."""
        fake_redis.setex = MagicMock(side_effect=Exception("connection lost"))
        cid = await mock_ipfs_cache.set("key1", {"data": "value"})
        assert cid is None
