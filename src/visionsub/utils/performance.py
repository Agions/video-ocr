"""
Performance optimization utilities for VisionSub
"""
import asyncio
import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import psutil

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry for OCR results"""
    key: str
    data: Any
    timestamp: float
    access_count: int = 0
    size_bytes: int = 0


class LRUCache:
    """Least Recently Used (LRU) cache implementation"""
    
    def __init__(self, max_size: int = 100, max_memory_mb: int = 512):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []
        self.total_memory_used = 0
        
    def _get_key(self, frame: np.ndarray, config_hash: str) -> str:
        """Generate cache key for frame"""
        # Create hash of frame data
        frame_hash = hashlib.md5(frame.tobytes()).hexdigest()
        return f"{frame_hash}_{config_hash}"
    
    def _calculate_memory_usage(self, data: Any) -> int:
        """Calculate memory usage of cached data"""
        try:
            if isinstance(data, np.ndarray):
                return data.nbytes
            elif isinstance(data, (list, tuple)):
                return sum(self._calculate_memory_usage(item) for item in data)
            elif isinstance(data, dict):
                return sum(self._calculate_memory_usage(v) for v in data.values())
            else:
                return len(str(data).encode('utf-8'))
        except:
            return 1024  # Fallback estimate
    
    def get(self, frame: np.ndarray, config_hash: str) -> Optional[Any]:
        """Get cached result"""
        key = self._get_key(frame, config_hash)
        
        if key in self.cache:
            entry = self.cache[key]
            entry.access_count += 1
            entry.timestamp = time.time()
            
            # Move to end of access order
            self.access_order.remove(key)
            self.access_order.append(key)
            
            return entry.data
        
        return None
    
    def put(self, frame: np.ndarray, config_hash: str, data: Any) -> bool:
        """Put result in cache"""
        key = self._get_key(frame, config_hash)
        data_size = self._calculate_memory_usage(data)
        
        # Check if single item exceeds memory limit
        if data_size > self.max_memory_bytes:
            logger.warning(f"Cache entry too large: {data_size} bytes")
            return False
        
        # Evict entries if necessary
        while (len(self.cache) >= self.max_size or 
               self.total_memory_used + data_size > self.max_memory_bytes):
            if not self._evict_oldest():
                break
        
        # Add new entry
        entry = CacheEntry(
            key=key,
            data=data,
            timestamp=time.time(),
            size_bytes=data_size
        )
        
        self.cache[key] = entry
        self.access_order.append(key)
        self.total_memory_used += data_size
        
        return True
    
    def _evict_oldest(self) -> bool:
        """Evict oldest entry from cache"""
        if not self.access_order:
            return False
        
        oldest_key = self.access_order.pop(0)
        if oldest_key in self.cache:
            entry = self.cache.pop(oldest_key)
            self.total_memory_used -= entry.size_bytes
            return True
        
        return False
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.access_order.clear()
        self.total_memory_used = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'memory_used_mb': self.total_memory_used / (1024 * 1024),
            'max_memory_mb': self.max_memory_bytes,
            'hit_rate': self._calculate_hit_rate(),
            'oldest_entry_age': self._get_oldest_entry_age()
        }
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        if not self.cache:
            return 0.0
        
        total_accesses = sum(entry.access_count for entry in self.cache.values())
        if total_accesses == 0:
            return 0.0
        
        # Access count includes initial puts, so hit rate is (accesses - size) / accesses
        hits = max(0, total_accesses - len(self.cache))
        return hits / total_accesses
    
    def _get_oldest_entry_age(self) -> float:
        """Get age of oldest cache entry in seconds"""
        if not self.cache:
            return 0.0
        
        oldest_time = min(entry.timestamp for entry in self.cache.values())
        return time.time() - oldest_time


class PerformanceMonitor:
    """Monitor system performance during processing"""
    
    def __init__(self):
        self.metrics = {
            'cpu_percent': [],
            'memory_percent': [],
            'processing_times': [],
            'frame_count': 0,
            'start_time': time.time()
        }
        self.monitoring = False
    
    def start_monitoring(self):
        """Start performance monitoring"""
        self.monitoring = True
        self.metrics['start_time'] = time.time()
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False
    
    def record_metrics(self):
        """Record current system metrics"""
        if not self.monitoring:
            return
        
        try:
            self.metrics['cpu_percent'].append(psutil.cpu_percent())
            self.metrics['memory_percent'].append(psutil.virtual_memory().percent)
        except Exception as e:
            logger.warning(f"Failed to record metrics: {e}")
    
    def record_frame_processing(self, processing_time: float):
        """Record frame processing time"""
        self.metrics['processing_times'].append(processing_time)
        self.metrics['frame_count'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.metrics['processing_times']:
            return {}
        
        total_time = time.time() - self.metrics['start_time']
        
        return {
            'total_processing_time': total_time,
            'frames_processed': self.metrics['frame_count'],
            'avg_frame_time': np.mean(self.metrics['processing_times']),
            'max_frame_time': np.max(self.metrics['processing_times']),
            'min_frame_time': np.min(self.metrics['processing_times']),
            'fps': self.metrics['frame_count'] / total_time if total_time > 0 else 0,
            'avg_cpu_percent': np.mean(self.metrics['cpu_percent']) if self.metrics['cpu_percent'] else 0,
            'avg_memory_percent': np.mean(self.metrics['memory_percent']) if self.metrics['memory_percent'] else 0,
            'peak_memory_percent': np.max(self.metrics['memory_percent']) if self.metrics['memory_percent'] else 0
        }


class ParallelProcessor:
    """Parallel processing utility for frame processing"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(8, (psutil.cpu_count() or 1) + 4)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        logger.info(f"Initialized parallel processor with {self.max_workers} workers")
    
    async def process_frames_parallel(
        self,
        frames: List[Tuple[np.ndarray, float]],
        process_func: Callable[[np.ndarray, float], Any],
        batch_size: int = 10
    ) -> List[Any]:
        """
        Process frames in parallel
        
        Args:
            frames: List of (frame, timestamp) tuples
            process_func: Function to process each frame
            batch_size: Number of frames to process in each batch
            
        Returns:
            List of processing results
        """
        results = []
        
        # Process frames in batches to avoid memory issues
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            logger.debug(f"Processing batch {i//batch_size + 1}/{(len(frames)-1)//batch_size + 1}")
            
            # Submit tasks to thread pool and track order
            future_to_info = {}
            frame_futures = []
            
            for frame, timestamp in batch:
                future = self.executor.submit(self._process_frame_sync, process_func, frame, timestamp)
                future_to_info[future] = {'frame': frame, 'timestamp': timestamp, 'index': len(results)}
                frame_futures.append(future)
                results.append(None)  # Placeholder for result
            
            # Wait for all futures to complete and update results in order
            for future in frame_futures:
                try:
                    result = future.result()
                    info = future_to_info[future]
                    results[info['index']] = result
                except Exception as e:
                    logger.error(f"Frame processing failed: {e}")
                    # Result is already None from placeholder
        
        return results
    
    def _process_frame_sync(
        self, 
        process_func: Callable[[np.ndarray, float], Any], 
        frame: np.ndarray, 
        timestamp: float
    ) -> Any:
        """Synchronous frame processing wrapper"""
        try:
            return process_func(frame, timestamp)
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            raise
    
    def shutdown(self):
        """Shutdown the thread pool"""
        self.executor.shutdown(wait=True)


def memory_efficient(func: Callable) -> Callable:
    """Decorator for memory-efficient processing"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Record memory before processing
        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024 * 1024)  # MB
        
        try:
            result = await func(*args, **kwargs)
            
            # Record memory after processing
            memory_after = process.memory_info().rss / (1024 * 1024)  # MB
            memory_diff = memory_after - memory_before
            
            logger.debug(f"Memory usage: {memory_before:.1f}MB → {memory_after:.1f}MB "
                        f"(Δ{memory_diff:+.1f}MB)")
            
            return result
            
        except Exception as e:
            logger.error(f"Function {func.__name__} failed: {e}")
            raise
    
    return wrapper


def timed_execution(func: Callable) -> Callable:
    """Decorator to time function execution"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} executed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {e}")
            raise
    
    return wrapper


class AdaptiveFrameSampler:
    """Adaptive frame sampling based on video characteristics"""
    
    def __init__(self, base_interval: float = 1.0, min_interval: float = 0.1, max_interval: float = 5.0):
        self.base_interval = base_interval
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.scene_changes = []
        self.frame_complexity = []
    
    def calculate_optimal_interval(
        self, 
        video_duration: float, 
        estimated_scene_changes: int = None,
        target_frame_count: int = 1000
    ) -> float:
        """Calculate optimal frame sampling interval"""
        if estimated_scene_changes:
            # More scene changes = more frequent sampling
            scene_density = estimated_scene_changes / video_duration
            adaptive_interval = self.base_interval / (1 + scene_density * 2)
        else:
            adaptive_interval = self.base_interval
        
        # Ensure interval is within bounds
        adaptive_interval = max(self.min_interval, min(self.max_interval, adaptive_interval))
        
        return adaptive_interval
    
    def should_sample_frame(
        self, 
        frame: np.ndarray, 
        timestamp: float, 
        last_sampled_time: float,
        scene_change_score: float = 0.0
    ) -> bool:
        """Determine if frame should be sampled"""
        # Always sample if enough time has passed
        if timestamp - last_sampled_time >= self.base_interval:
            return True
        
        # Sample on scene changes
        if scene_change_score > 0.5:  # Significant scene change
            return True
        
        # Sample based on frame complexity (motion, text content)
        complexity_score = self._calculate_frame_complexity(frame)
        if complexity_score > 0.7:  # High complexity
            return True
        
        return False
    
    def _calculate_frame_complexity(self, frame: np.ndarray) -> float:
        """Calculate frame complexity score"""
        try:
            # Convert to grayscale
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # Calculate edge density (proxy for complexity)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Calculate text region density (simplified)
            # This is a proxy - actual text detection would be more accurate
            text_regions = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            text_density = np.sum(text_regions > 0) / text_regions.size
            
            # Combine scores
            complexity = (edge_density * 0.6 + text_density * 0.4)
            return min(1.0, complexity)
            
        except Exception as e:
            logger.warning(f"Failed to calculate frame complexity: {e}")
            return 0.0