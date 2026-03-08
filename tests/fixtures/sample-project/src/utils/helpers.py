# Utility helpers for caching and hashing
# Provides content-addressable cache management

import hashlib
from typing import Dict, Optional


def hash_content(text: str) -> str:
    """Generate a content hash for cache invalidation."""
    return hashlib.sha256(text.encode()).hexdigest()[:12]


class CacheManager:
    """Simple in-memory cache with TTL support."""

    def __init__(self, max_size: int = 1000):
        self._cache: Dict[str, tuple] = {}
        self._max_size = max_size

    def get(self, key: str) -> Optional[str]:
        if key in self._cache:
            value, _ts = self._cache[key]
            return value
        return None

    def set(self, key: str, value: str) -> None:
        if len(self._cache) >= self._max_size:
            oldest = min(self._cache, key=lambda k: self._cache[k][1])
            del self._cache[oldest]
        self._cache[key] = (value, __import__('time').time())

    def invalidate(self, key: str) -> bool:
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)
