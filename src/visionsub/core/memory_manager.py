"""
Enhanced memory management and image processing cache system
"""
import asyncio
import gc
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from weakref import WeakKeyDictionary, WeakValueDictionary

import numpy as np
from PIL import Image

from ..core.errors import VisionSubError, ErrorCategory, ErrorSeverity


class CacheStrategy(Enum):
    """Cache eviction strategies"""
    LRU = "lru"           # Least Recently Used
    FIFO = "fifo"         # First In First Out
    LFU = "lfu"           # Least Frequently Used
    ADAPTIVE = "adaptive" # Adaptive based on access patterns


class CompressionLevel(Enum):
    """Image compression levels"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    data: Any
    timestamp: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    size_bytes: int = 0
    compression_level: CompressionLevel = CompressionLevel.NONE
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryStats:
    """Memory usage statistics"""
    total_memory_mb: float = 0.0
    used_memory_mb: float = 0.0
    cache_memory_mb: float = 0.0
    available_memory_mb: float = 0.0
    cache_hit_rate: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    eviction_count: int = 0
    compression_ratio: float = 1.0


class ImageCache:
    """Intelligent image cache with compression and memory management"""
    
    def __init__(
        self, 
        max_size_mb: float = 512.0,
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
        compression_level: CompressionLevel = CompressionLevel.MEDIUM,
        enable_compression: bool = True
    ):
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.strategy = strategy
        self.compression_level = compression_level
        self.enable_compression = enable_compression
        
        # Cache storage
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []  # For LRU/FIFO
        self.access_frequency: Dict[str, int] = {}  # For LFU
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = MemoryStats()
        self._stats_lock = threading.Lock()
        
        # Memory monitoring
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Compression
        self.compressor = ImageCompressor()
        
        self.logger = logging.getLogger(__name__)

    def put(self, key: str, data: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add data to cache"""
        try:
            # Calculate size
            size_bytes = self._calculate_size(data)
            
            # Check if we need to evict
            with self._lock:
                if size_bytes > self.max_size_bytes:
                    self.logger.warning(f"Data size {size_bytes} bytes exceeds cache max size {self.max_size_bytes} bytes")
                    return False
                
                self._ensure_space(size_bytes)
                
                # Compress if enabled and it's an image
                compressed_data = data
                actual_compression = CompressionLevel.NONE
                
                if self.enable_compression and isinstance(data, np.ndarray):
                    compressed_data, actual_compression = self.compressor.compress(
                        data, self.compression_level
                    )
                    size_bytes = self._calculate_size(compressed_data)
                
                # Create cache entry
                entry = CacheEntry(
                    key=key,
                    data=compressed_data,
                    timestamp=time.time(),
                    size_bytes=size_bytes,
                    compression_level=actual_compression,
                    metadata=metadata or {}
                )
                
                # Remove existing entry if present
                if key in self.cache:
                    self._remove_entry(key)
                
                # Add new entry
                self.cache[key] = entry
                self._update_access_info(key)
                
                # Update statistics
                self._update_stats()
                
                self.logger.debug(f"Added to cache: {key} ({size_bytes} bytes)")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to add to cache: {e}")
            return False

    def get(self, key: str) -> Optional[Any]:
        """Get data from cache"""
        with self._lock:
            if key not in self.cache:
                with self._stats_lock:
                    self.stats.cache_misses += 1
                return None
            
            entry = self.cache[key]
            
            # Update access info
            entry.access_count += 1
            entry.last_access = time.time()
            self._update_access_info(key)
            
            # Decompress if needed
            data = entry.data
            if entry.compression_level != CompressionLevel.NONE:
                data = self.compressor.decompress(data, entry.compression_level)
            
            # Update statistics
            with self._stats_lock:
                self.stats.cache_hits += 1
            
            self.logger.debug(f"Cache hit: {key}")
            return data

    def remove(self, key: str) -> bool:
        """Remove entry from cache"""
        with self._lock:
            if key in self.cache:
                self._remove_entry(key)
                self._update_stats()
                return True
            return False

    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self.cache.clear()
            self.access_order.clear()
            self.access_frequency.clear()
            self._update_stats()
            gc.collect()

    def _remove_entry(self, key: str):
        """Remove entry from cache and auxiliary structures"""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_order:
            self.access_order.remove(key)
        if key in self.access_frequency:
            del self.access_frequency[key]

    def _update_access_info(self, key: str):
        """Update access information for eviction strategies"""
        # Update access order for LRU/FIFO
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
        
        # Update frequency for LFU
        self.access_frequency[key] = self.access_frequency.get(key, 0) + 1

    def _ensure_space(self, required_bytes: int):
        """Ensure there's enough space by evicting entries"""
        current_size = sum(entry.size_bytes for entry in self.cache.values())
        available_space = self.max_size_bytes - current_size
        
        if available_space >= required_bytes:
            return
        
        # Need to evict
        to_evict = self._select_entries_to_evict(required_bytes - available_space)
        
        for key in to_evict:
            self._remove_entry(key)
            with self._stats_lock:
                self.stats.eviction_count += 1

    def _select_entries_to_evict(self, required_bytes: int) -> List[str]:
        """Select entries to evict based on strategy"""
        if not self.cache:
            return []
        
        if self.strategy == CacheStrategy.LRU:
            # Least Recently Used
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: x[1].last_access
            )
        elif self.strategy == CacheStrategy.FIFO:
            # First In First Out
            sorted_entries = [
                (key, self.cache[key]) 
                for key in self.access_order
            ]
        elif self.strategy == CacheStrategy.LFU:
            # Least Frequently Used
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: x[1].access_count
            )
        else:  # ADAPTIVE
            # Adaptive: Combine LRU and LFU with size consideration
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: (
                    x[1].last_access * 0.3 + 
                    (1.0 / (x[1].access_count + 1)) * 0.4 +
                    x[1].size_bytes * 0.3
                )
            )
        
        # Select entries to evict
        to_evict = []
        freed_bytes = 0
        
        for key, entry in sorted_entries:
            to_evict.append(key)
            freed_bytes += entry.size_bytes
            
            if freed_bytes >= required_bytes:
                break
        
        return to_evict

    def _calculate_size(self, data: Any) -> int:
        """Calculate size of data in bytes"""
        if isinstance(data, np.ndarray):
            return data.nbytes
        elif isinstance(data, (str, bytes)):
            return len(data.encode('utf-8')) if isinstance(data, str) else len(data)
        else:
            # Rough estimate for other types
            return len(str(data).encode('utf-8'))

    def _update_stats(self):
        """Update memory statistics"""
        try:
            import psutil
            
            # Get system memory info
            memory = psutil.virtual_memory()
            
            with self._stats_lock:
                self.stats.total_memory_mb = memory.total / (1024 * 1024)
                self.stats.used_memory_mb = memory.used / (1024 * 1024)
                self.stats.available_memory_mb = memory.available / (1024 * 1024)
                
                # Calculate cache memory
                cache_memory = sum(entry.size_bytes for entry in self.cache.values())
                self.stats.cache_memory_mb = cache_memory / (1024 * 1024)
                
                # Calculate hit rate
                total_accesses = self.stats.cache_hits + self.stats.cache_misses
                if total_accesses > 0:
                    self.stats.cache_hit_rate = self.stats.cache_hits / total_accesses
                
                # Calculate compression ratio
                original_size = sum(
                    self._calculate_original_size(entry) 
                    for entry in self.cache.values()
                )
                compressed_size = sum(entry.size_bytes for entry in self.cache.values())
                
                if original_size > 0:
                    self.stats.compression_ratio = original_size / compressed_size
                    
        except ImportError:
            # psutil not available, use basic stats
            cache_memory = sum(entry.size_bytes for entry in self.cache.values())
            with self._stats_lock:
                self.stats.cache_memory_mb = cache_memory / (1024 * 1024)

    def _calculate_original_size(self, entry: CacheEntry) -> int:
        """Calculate original size before compression"""
        if entry.compression_level == CompressionLevel.NONE:
            return entry.size_bytes
        
        # Estimate original size based on compression level
        compression_ratios = {
            CompressionLevel.LOW: 0.7,
            CompressionLevel.MEDIUM: 0.5,
            CompressionLevel.HIGH: 0.3,
            CompressionLevel.MAXIMUM: 0.2
        }
        
        ratio = compression_ratios.get(entry.compression_level, 1.0)
        return int(entry.size_bytes / ratio)

    def get_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        self._update_stats()
        with self._stats_lock:
            return self.stats

    def start_monitoring(self, interval_seconds: float = 5.0):
        """Start memory monitoring"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_memory,
            daemon=True
        )
        self._monitor_thread.start()
        self.logger.info("Started memory monitoring")

    def stop_monitoring(self):
        """Stop memory monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
        self.logger.info("Stopped memory monitoring")

    def _monitor_memory(self):
        """Memory monitoring loop"""
        while self._monitoring:
            try:
                self._update_stats()
                
                # Check memory pressure
                if self.stats.used_memory_mb > self.stats.total_memory_mb * 0.9:
                    self.logger.warning("High memory pressure detected, forcing cleanup")
                    self._force_cleanup()
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Memory monitoring error: {e}")
                break

    def _force_cleanup(self):
        """Force cleanup of cache"""
        with self._lock:
            # Remove least recently used entries
            if self.cache:
                # Remove 25% of entries
                num_to_remove = max(1, len(self.cache) // 4)
                for _ in range(num_to_remove):
                    if self.access_order:
                        key = self.access_order.pop(0)
                        self._remove_entry(key)
            
            # Force garbage collection
            gc.collect()


class ImageCompressor:
    """Image compression utility"""
    
    def __init__(self):
        self.compression_settings = {
            CompressionLevel.LOW: {'quality': 90, 'optimize': True},
            CompressionLevel.MEDIUM: {'quality': 75, 'optimize': True},
            CompressionLevel.HIGH: {'quality': 50, 'optimize': True},
            CompressionLevel.MAXIMUM: {'quality': 30, 'optimize': True}
        }

    def compress(self, image: np.ndarray, level: CompressionLevel) -> Tuple[Any, CompressionLevel]:
        """Compress image data"""
        if level == CompressionLevel.NONE:
            return image, CompressionLevel.NONE
        
        try:
            # Convert numpy array to PIL Image
            if len(image.shape) == 3:
                # Color image
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                # Grayscale image
                pil_image = Image.fromarray(image)
            
            # Compress to bytes
            settings = self.compression_settings[level]
            img_bytes = io.BytesIO()
            pil_image.save(img_bytes, format='JPEG', **settings)
            compressed_data = img_bytes.getvalue()
            
            return compressed_data, level
            
        except Exception as e:
            logging.warning(f"Compression failed, using original: {e}")
            return image, CompressionLevel.NONE

    def decompress(self, data: Any, level: CompressionLevel) -> np.ndarray:
        """Decompress image data"""
        if level == CompressionLevel.NONE:
            return data
        
        try:
            if isinstance(data, bytes):
                # Convert bytes back to image
                img_bytes = io.BytesIO(data)
                pil_image = Image.open(img_bytes)
                
                # Convert back to numpy array
                image = np.array(pil_image)
                
                # Convert RGB to BGR if needed
                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                return image
            else:
                return data
                
        except Exception as e:
            logging.warning(f"Decompression failed, returning original: {e}")
            return data


class MemoryPool:
    """Memory pool for efficient allocation of large objects"""
    
    def __init__(self, pool_size: int = 10, max_size_mb: float = 100.0):
        self.pool_size = pool_size
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.pools: Dict[str, List] = {}
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)

    def allocate(self, shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        """Allocate numpy array from pool"""
        pool_key = f"{shape}_{dtype}"
        
        with self._lock:
            if pool_key in self.pools and self.pools[pool_key]:
                # Get from pool
                array = self.pools[pool_key].pop()
                array.fill(0)  # Clear array
                return array
            
        # Create new array
        try:
            array = np.zeros(shape, dtype=dtype)
            
            # Check size and add to pool if small enough
            if array.nbytes < self.max_size_bytes / self.pool_size:
                with self._lock:
                    if pool_key not in self.pools:
                        self.pools[pool_key] = []
                    
                    if len(self.pools[pool_key]) < self.pool_size:
                        self.pools[pool_key].append(array)
            
            return array
            
        except MemoryError:
            # Try to free some memory
            self._free_memory()
            return np.zeros(shape, dtype=dtype)

    def _free_memory(self):
        """Free memory by clearing pools"""
        with self._lock:
            for pool_key in list(self.pools.keys()):
                if len(self.pools[pool_key]) > 0:
                    self.pools[pool_key].clear()
                    self.logger.info(f"Cleared memory pool: {pool_key}")
        
        gc.collect()

    def clear(self):
        """Clear all pools"""
        with self._lock:
            for pool in self.pools.values():
                pool.clear()
            self.pools.clear()
        
        gc.collect()


class MemoryManager:
    """Central memory management system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        self.image_cache = ImageCache(
            max_size_mb=config.get('cache_size_mb', 512),
            strategy=CacheStrategy.ADAPTIVE,
            compression_level=CompressionLevel.MEDIUM,
            enable_compression=config.get('enable_compression', True)
        )
        
        self.memory_pool = MemoryPool(
            pool_size=config.get('pool_size', 10),
            max_size_mb=config.get('pool_max_size_mb', 100)
        )
        
        self.logger = logging.getLogger(__name__)
        
        # Start monitoring
        self.image_cache.start_monitoring()

    def get_image_cache(self) -> ImageCache:
        """Get image cache instance"""
        return self.image_cache

    def get_memory_pool(self) -> MemoryPool:
        """Get memory pool instance"""
        return self.memory_pool

    def cleanup(self):
        """Cleanup resources"""
        self.image_cache.stop_monitoring()
        self.image_cache.clear()
        self.memory_pool.clear()

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        cache_stats = self.image_cache.get_stats()
        
        return {
            'cache_stats': cache_stats,
            'pool_info': {
                'total_pools': len(self.memory_pool.pools),
                'total_pooled_items': sum(len(pool) for pool in self.memory_pool.pools.values())
            },
            'config': self.config
        }

    def __del__(self):
        """Destructor"""
        try:
            self.cleanup()
        except:
            pass


# Global memory manager instance
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get global memory manager instance"""
    global _memory_manager
    if _memory_manager is None:
        config = {
            'cache_size_mb': 512,
            'enable_compression': True,
            'pool_size': 10,
            'pool_max_size_mb': 100
        }
        _memory_manager = MemoryManager(config)
    return _memory_manager


def initialize_memory_manager(config: Dict[str, Any]):
    """Initialize global memory manager with custom config"""
    global _memory_manager
    _memory_manager = MemoryManager(config)


# Decorator for memory-efficient processing
def memory_efficient(max_size_mb: float = 100.0):
    """Decorator to make functions memory-efficient"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get memory manager
            memory_manager = get_memory_manager()
            
            # Get initial memory usage
            initial_stats = memory_manager.get_memory_stats()
            initial_memory = initial_stats['cache_stats'].used_memory_mb
            
            try:
                result = func(*args, **kwargs)
                
                # Check memory usage after execution
                final_stats = memory_manager.get_memory_stats()
                final_memory = final_stats['cache_stats'].used_memory_mb
                
                memory_increase = final_memory - initial_memory
                
                if memory_increase > max_size_mb:
                    logging.warning(f"Function {func.__name__} used {memory_increase:.2f}MB memory")
                    # Force cleanup
                    memory_manager.image_cache._force_cleanup()
                
                return result
                
            except MemoryError:
                logging.error(f"Memory error in {func.__name__}, attempting cleanup")
                memory_manager.image_cache._force_cleanup()
                raise
        
        return wrapper
    return decorator


# Import required modules
try:
    import cv2
except ImportError:
    cv2 = None

try:
    import io
except ImportError:
    io = None

try:
    import psutil
except ImportError:
    psutil = None