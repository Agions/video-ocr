"""
Metrics Collection and Monitoring Utilities
"""
import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: float
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricStats:
    """Metric statistics"""
    count: int = 0
    sum: float = 0.0
    min: float = float('inf')
    max: float = float('-inf')
    avg: float = 0.0
    p95: float = 0.0
    p99: float = 0.0


class MetricsCollector:
    """Enhanced metrics collector with various metric types"""

    def __init__(self, max_history_size: int = 1000):
        self.max_history_size = max_history_size
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history_size))
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._timers: Dict[str, List[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def record_metric(self, name: str, value: float, metadata: Optional[Dict[str, Any]] = None):
        """Record a metric value"""
        async with self._lock:
            metric_point = MetricPoint(
                timestamp=time.time(),
                value=value,
                metadata=metadata or {}
            )
            self._metrics[name].append(metric_point)

    async def increment_counter(self, name: str, value: int = 1):
        """Increment a counter"""
        async with self._lock:
            self._counters[name] += value

    async def set_gauge(self, name: str, value: float):
        """Set a gauge value"""
        async with self._lock:
            self._gauges[name] = value

    async def record_timing(self, name: str, duration: float):
        """Record a timing measurement"""
        async with self._lock:
            self._timers[name].append(duration)
            
            # Keep only recent measurements
            if len(self._timers[name]) > self.max_history_size:
                self._timers[name] = self._timers[name][-self.max_history_size:]

    async def get_metric_stats(self, name: str) -> Optional[MetricStats]:
        """Get statistics for a metric"""
        async with self._lock:
            if name not in self._metrics or not self._metrics[name]:
                return None
            
            values = [point.value for point in self._metrics[name]]
            if not values:
                return None
            
            stats = MetricStats(
                count=len(values),
                sum=sum(values),
                min=min(values),
                max=max(values),
                avg=sum(values) / len(values)
            )
            
            # Calculate percentiles
            sorted_values = sorted(values)
            if len(sorted_values) >= 20:
                stats.p95 = sorted_values[int(len(sorted_values) * 0.95)]
                stats.p99 = sorted_values[int(len(sorted_values) * 0.99)]
            
            return stats

    async def get_counter(self, name: str) -> int:
        """Get counter value"""
        async with self._lock:
            return self._counters[name]

    async def get_gauge(self, name: str) -> Optional[float]:
        """Get gauge value"""
        async with self._lock:
            return self._gauges.get(name)

    async def get_timing_stats(self, name: str) -> Optional[MetricStats]:
        """Get timing statistics"""
        async with self._lock:
            if name not in self._timers or not self._timers[name]:
                return None
            
            values = self._timers[name]
            stats = MetricStats(
                count=len(values),
                sum=sum(values),
                min=min(values),
                max=max(values),
                avg=sum(values) / len(values)
            )
            
            # Calculate percentiles
            sorted_values = sorted(values)
            if len(sorted_values) >= 20:
                stats.p95 = sorted_values[int(len(sorted_values) * 0.95)]
                stats.p99 = sorted_values[int(len(sorted_values) * 0.99)]
            
            return stats

    async def get_recent_metrics(self, name: str, count: int = 100) -> List[MetricPoint]:
        """Get recent metric points"""
        async with self._lock:
            if name not in self._metrics:
                return []
            
            return list(self._metrics[name])[-count:]

    async def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics"""
        async with self._lock:
            result = {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "metrics_summary": {}
            }
            
            # Add metric summaries
            for name in self._metrics:
                stats = await self.get_metric_stats(name)
                if stats:
                    result["metrics_summary"][name] = stats.__dict__
            
            # Add timing summaries
            for name in self._timers:
                stats = await self.get_timing_stats(name)
                if stats:
                    result["metrics_summary"][f"{name}_timing"] = stats.__dict__
            
            return result

    async def reset_metrics(self):
        """Reset all metrics"""
        async with self._lock:
            self._metrics.clear()
            self._counters.clear()
            self._gauges.clear()
            self._timers.clear()

    # Convenience methods
    async def record_processing_time(self, duration: float):
        """Record processing time"""
        await self.record_timing("processing_time", duration)

    async def record_frame_processing_time(self, duration: float):
        """Record frame processing time"""
        await self.record_timing("frame_processing_time", duration)

    async def record_ocr_processing_time(self, duration: float):
        """Record OCR processing time"""
        await self.record_timing("ocr_processing_time", duration)

    async def increment_frames_processed(self):
        """Increment frames processed counter"""
        await self.increment_counter("frames_processed")

    async def increment_total_scene_changes(self):
        """Increment scene changes counter"""
        await self.increment_counter("scene_changes")

    async def increment_error_count(self):
        """Increment error counter"""
        await self.increment_counter("errors")

    async def get_total_frames_processed(self) -> int:
        """Get total frames processed"""
        return await self.get_counter("frames_processed") or 0

    async def get_total_scene_changes(self) -> int:
        """Get total scene changes"""
        return await self.get_counter("scene_changes") or 0

    async def get_average_processing_time(self) -> float:
        """Get average processing time"""
        stats = await self.get_timing_stats("processing_time")
        return stats.avg if stats else 0.0

    async def get_error_count(self) -> int:
        """Get error count"""
        return await self.get_counter("errors") or 0

    async def get_success_rate(self) -> float:
        """Calculate success rate"""
        total = await self.get_counter("frames_processed") or 0
        errors = await self.get_counter("errors") or 0
        
        if total == 0:
            return 100.0
        
        return max(0.0, ((total - errors) / total) * 100.0)

    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics"""
        return await self.get_all_metrics()


class PerformanceMonitor:
    """Performance monitoring for video processing"""

    def __init__(self, max_window_size: int = 1000):
        self.max_window_size = max_window_size
        self._frame_times: deque = deque(maxlen=max_window_size)
        self._memory_usage: deque = deque(maxlen=max_window_size)
        self._cpu_usage: deque = deque(maxlen=max_window_size)
        self._start_time = time.time()
        self._total_frames_processed = 0

    def record_frame_processed(self, processing_time: float, memory_usage: float):
        """Record frame processing metrics"""
        self._frame_times.append(processing_time)
        self._memory_usage.append(memory_usage)
        self._total_frames_processed += 1

    def get_average_frame_time(self) -> float:
        """Get average frame processing time"""
        if not self._frame_times:
            return 0.0
        return sum(self._frame_times) / len(self._frame_times)

    def get_peak_memory_usage(self) -> float:
        """Get peak memory usage"""
        if not self._memory_usage:
            return 0.0
        return max(self._memory_usage)

    def get_average_memory_usage(self) -> float:
        """Get average memory usage"""
        if not self._memory_usage:
            return 0.0
        return sum(self._memory_usage) / len(self._memory_usage)

    def get_throughput_fps(self) -> float:
        """Get processing throughput in FPS"""
        uptime = time.time() - self._start_time
        if uptime == 0:
            return 0.0
        return self._total_frames_processed / uptime

    def get_recent_performance(self, window_size: int = 100) -> Dict[str, float]:
        """Get recent performance metrics"""
        recent_frames = list(self._frame_times)[-window_size:]
        recent_memory = list(self._memory_usage)[-window_size:]
        
        return {
            "average_frame_time": sum(recent_frames) / len(recent_frames) if recent_frames else 0.0,
            "peak_memory": max(recent_memory) if recent_memory else 0.0,
            "average_memory": sum(recent_memory) / len(recent_memory) if recent_memory else 0.0,
            "frames_processed": len(recent_frames)
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        return {
            "uptime": time.time() - self._start_time,
            "total_frames_processed": self._total_frames_processed,
            "average_frame_time": self.get_average_frame_time(),
            "peak_memory_usage": self.get_peak_memory_usage(),
            "average_memory_usage": self.get_average_memory_usage(),
            "throughput_fps": self.get_throughput_fps(),
            "recent_performance": self.get_recent_performance()
        }


class HealthMonitor:
    """Health monitoring for system components"""

    def __init__(self):
        self._health_checks: Dict[str, Callable] = {}
        self._health_status: Dict[str, bool] = {}
        self._last_check_time: Dict[str, float] = {}
        self._check_interval = 60.0  # seconds

    def register_check(self, name: str, check_func: Callable[[], bool]):
        """Register a health check function"""
        self._health_checks[name] = check_func
        self._health_status[name] = True
        self._last_check_time[name] = 0.0

    async def perform_health_checks(self) -> Dict[str, bool]:
        """Perform all registered health checks"""
        results = {}
        
        for name, check_func in self._health_checks.items():
            try:
                current_time = time.time()
                
                # Skip if checked recently
                if current_time - self._last_check_time[name] < self._check_interval:
                    results[name] = self._health_status[name]
                    continue
                
                # Perform health check
                if asyncio.iscoroutinefunction(check_func):
                    is_healthy = await check_func()
                else:
                    is_healthy = check_func()
                
                self._health_status[name] = is_healthy
                self._last_check_time[name] = current_time
                results[name] = is_healthy
                
            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                self._health_status[name] = False
                results[name] = False
        
        return results

    async def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        health_results = await self.perform_health_checks()
        
        overall_healthy = all(health_results.values())
        
        return {
            "healthy": overall_healthy,
            "components": health_results,
            "timestamp": time.time(),
            "check_interval": self._check_interval
        }

    async def cleanup(self):
        """Cleanup health monitor"""
        self._health_checks.clear()
        self._health_status.clear()
        self._last_check_time.clear()