"""
Frame caching system for optimizing repeated OCR operations
"""
import hashlib
import logging
import time
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class CacheEntry:
    """Represents a single cache entry"""

    def __init__(self, result: Any, frame_hash: str):
        self.result = result
        self.frame_hash = frame_hash
        self.timestamp = time.time()
        self.access_count = 1

    def update_access(self):
        """Update access information"""
        self.timestamp = time.time()
        self.access_count += 1

    def get_age(self) -> float:
        """Get age of cache entry in seconds"""
        return time.time() - self.timestamp

    def get_score(self) -> float:
        """Get cache score for eviction decisions"""
        # Score based on access count and age
        age_factor = 1.0 / (1.0 + self.get_age())
        return self.access_count * age_factor


class FrameCache:
    """
    LRU (Least Recently Used) cache for frame processing results
    """

    def __init__(self, max_size: int = 100, max_age: float = 300.0):
        """
        Initialize frame cache

        Args:
            max_size: Maximum number of entries in cache
            max_age: Maximum age of cache entries in seconds
        """
        self.max_size = max_size
        self.max_age = max_age
        self._cache: Dict[str, CacheEntry] = {}
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str) -> Optional[Any]:
        """
        Get result from cache

        Args:
            key: Cache key

        Returns:
            Cached result or None if not found
        """
        entry = self._cache.get(key)

        if entry is None:
            self._misses += 1
            return None

        # Check if entry has expired
        if entry.get_age() > self.max_age:
            self._remove_entry(key)
            self._misses += 1
            return None

        # Update access information
        entry.update_access()
        self._hits += 1

        logger.debug(f"Cache hit for key: {key}")
        return entry.result

    def set(self, key: str, result: Any) -> None:
        """
        Store result in cache

        Args:
            key: Cache key
            result: Result to cache
        """
        # Check if cache is full
        if len(self._cache) >= self.max_size:
            self._evict_lru()

        # Create cache entry
        frame_hash = self._extract_frame_hash(key)
        entry = CacheEntry(result, frame_hash)
        self._cache[key] = entry

        logger.debug(f"Cache set for key: {key}")

    def _extract_frame_hash(self, key: str) -> str:
        """Extract frame hash from cache key"""
        try:
            # Key format: "frame_idx_hash"
            parts = key.split('_')
            if len(parts) >= 2:
                return parts[-1]
            return key
        except Exception:
            return key

    def _evict_lru(self) -> None:
        """Evict least recently used entry from cache"""
        if not self._cache:
            return

        # Find entry with lowest score
        lru_key = min(self._cache.keys(), key=lambda k: self._cache[k].get_score())
        self._remove_entry(lru_key)
        self._evictions += 1

    def _remove_entry(self, key: str) -> None:
        """Remove entry from cache"""
        if key in self._cache:
            del self._cache[key]

    def clear(self) -> None:
        """Clear all entries from cache"""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': hit_rate,
            'evictions': self._evictions,
            'max_age': self.max_age
        }

    def cleanup_expired(self) -> None:
        """Remove expired entries from cache"""
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.get_age() > self.max_age
        ]

        for key in expired_keys:
            self._remove_entry(key)

        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")


class CacheKeyGenerator:
    """Generates cache keys for frames"""

    @staticmethod
    def generate_key(
        frame: np.ndarray,
        frame_idx: int,
        roi_rect: tuple = None
    ) -> str:
        """
        Generate cache key for frame

        Args:
            frame: Video frame as numpy array
            frame_idx: Frame index
            roi_rect: Region of interest rectangle (x, y, w, h)

        Returns:
            Cache key string
        """
        # Create hash of frame content
        frame_hash = hashlib.md5(frame.tobytes()).hexdigest()[:8]

        # Include ROI information if provided
        roi_suffix = ""
        if roi_rect:
            roi_hash = hashlib.md5(str(roi_rect).encode()).hexdigest()[:4]
            roi_suffix = f"_{roi_hash}"

        return f"{frame_idx}_{frame_hash}{roi_suffix}"

    @staticmethod
    def generate_key_for_config(
        frame: np.ndarray,
        frame_idx: int,
        config: Dict[str, Any]
    ) -> str:
        """
        Generate cache key that includes processing configuration

        Args:
            frame: Video frame as numpy array
            frame_idx: Frame index
            config: Processing configuration

        Returns:
            Cache key string
        """
        # Create hash of configuration
        config_str = str(sorted(config.items()))
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:6]

        # Generate frame hash
        frame_hash = hashlib.md5(frame.tobytes()).hexdigest()[:8]

        return f"{frame_idx}_{frame_hash}_{config_hash}"
