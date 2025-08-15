"""
Structured logging and performance monitoring system
"""
import asyncio
import json
import logging
import logging.handlers
import os
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from uuid import uuid4

from ..models.config import LoggingConfig


class LogLevel(Enum):
    """Log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class MetricType(Enum):
    """Metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class LogContext:
    """Structured log context"""
    component: str
    operation: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    trace_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: datetime
    level: LogLevel
    message: str
    context: LogContext
    exception_info: Optional[Dict[str, Any]] = None
    performance_data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'level': self.level.value,
            'message': self.message,
            'context': asdict(self.context),
            'exception_info': self.exception_info,
            'performance_data': self.performance_data
        }


@dataclass
class Metric:
    """Performance metric"""
    name: str
    type: MetricType
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceSnapshot:
    """Performance snapshot at a point in time"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    network_io: Dict[str, int]
    process_cpu_percent: float
    process_memory_mb: float
    thread_count: int
    handle_count: Optional[int] = None


class StructuredLogger:
    """Structured logger with context awareness"""
    
    def __init__(self, name: str, config: LoggingConfig):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(name)
        
        # Setup logger
        self._setup_logger()
        
        # Default context
        self.default_context: Optional[LogContext] = None
        
    def _setup_logger(self):
        """Setup logger with handlers"""
        self.logger.setLevel(getattr(logging, self.config.level))
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Console handler
        if self.config.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, self.config.level))
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if self.config.log_file:
            log_file = Path(self.config.log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.config.max_file_size_mb * 1024 * 1024,
                backupCount=self.config.backup_count
            )
            file_handler.setLevel(getattr(logging, self.config.level))
            
            # JSON formatter for structured logging
            file_formatter = logging.Formatter(
                '%(message)s'  # We'll format the message as JSON
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def set_default_context(self, context: LogContext):
        """Set default context for all log entries"""
        self.default_context = context
    
    def _log(self, level: LogLevel, message: str, context: Optional[LogContext] = None, 
             exception: Optional[Exception] = None, **kwargs):
        """Internal logging method"""
        # Merge contexts
        final_context = self.default_context or LogContext("unknown", "unknown")
        if context:
            final_context.metadata.update(context.metadata)
            if context.component != "unknown":
                final_context.component = context.component
            if context.operation != "unknown":
                final_context.operation = context.operation
        
        # Add additional metadata
        final_context.metadata.update(kwargs)
        
        # Create log entry
        entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            message=message,
            context=final_context
        )
        
        # Add exception info if present
        if exception:
            entry.exception_info = {
                'type': type(exception).__name__,
                'message': str(exception),
                'traceback': self._get_traceback(exception)
            }
        
        # Log to Python logger
        log_message = json.dumps(entry.to_dict())
        getattr(self.logger, level.value.lower())(log_message)
    
    def _get_traceback(self, exception: Exception) -> str:
        """Get exception traceback"""
        import traceback
        return ''.join(traceback.format_exception(
            type(exception), exception, exception.__traceback__
        ))
    
    def debug(self, message: str, context: Optional[LogContext] = None, **kwargs):
        """Log debug message"""
        self._log(LogLevel.DEBUG, message, context, **kwargs)
    
    def info(self, message: str, context: Optional[LogContext] = None, **kwargs):
        """Log info message"""
        self._log(LogLevel.INFO, message, context, **kwargs)
    
    def warning(self, message: str, context: Optional[LogContext] = None, **kwargs):
        """Log warning message"""
        self._log(LogLevel.WARNING, message, context, **kwargs)
    
    def error(self, message: str, exception: Optional[Exception] = None, 
              context: Optional[LogContext] = None, **kwargs):
        """Log error message"""
        self._log(LogLevel.ERROR, message, context, exception, **kwargs)
    
    def critical(self, message: str, exception: Optional[Exception] = None, 
                 context: Optional[LogContext] = None, **kwargs):
        """Log critical message"""
        self._log(LogLevel.CRITICAL, message, context, exception, **kwargs)


class MetricsCollector:
    """Performance metrics collector"""
    
    def __init__(self):
        self.metrics: List[Metric] = []
        self.counters: Dict[str, float] = {}
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = {}
        self.timers: Dict[str, List[float]] = {}
        
        self._lock = threading.Lock()
        
    def counter(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """Increment or set counter"""
        with self._lock:
            self.counters[name] = self.counters.get(name, 0.0) + value
            
            metric = Metric(
                name=name,
                type=MetricType.COUNTER,
                value=self.counters[name],
                timestamp=datetime.now(),
                tags=tags or {}
            )
            self.metrics.append(metric)
    
    def gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set gauge value"""
        with self._lock:
            self.gauges[name] = value
            
            metric = Metric(
                name=name,
                type=MetricType.GAUGE,
                value=value,
                timestamp=datetime.now(),
                tags=tags or {}
            )
            self.metrics.append(metric)
    
    def histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Add value to histogram"""
        with self._lock:
            if name not in self.histograms:
                self.histograms[name] = []
            
            self.histograms[name].append(value)
            
            # Keep only recent values (last 1000)
            if len(self.histograms[name]) > 1000:
                self.histograms[name] = self.histograms[name][-1000:]
            
            metric = Metric(
                name=name,
                type=MetricType.HISTOGRAM,
                value=value,
                timestamp=datetime.now(),
                tags=tags or {}
            )
            self.metrics.append(metric)
    
    def timer(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Add timing value"""
        with self._lock:
            if name not in self.timers:
                self.timers[name] = []
            
            self.timers[name].append(value)
            
            # Keep only recent values (last 1000)
            if len(self.timers[name]) > 1000:
                self.timers[name] = self.timers[name][-1000:]
            
            metric = Metric(
                name=name,
                type=MetricType.TIMER,
                value=value,
                timestamp=datetime.now(),
                tags=tags or {}
            )
            self.metrics.append(metric)
    
    def get_metrics(self, metric_type: Optional[MetricType] = None, 
                   name_filter: Optional[str] = None) -> List[Metric]:
        """Get metrics with optional filtering"""
        with self._lock:
            filtered = self.metrics
            
            if metric_type:
                filtered = [m for m in filtered if m.type == metric_type]
            
            if name_filter:
                filtered = [m for m in filtered if name_filter in m.name]
            
            return filtered.copy()
    
    def get_stats(self, name: str) -> Dict[str, Any]:
        """Get statistics for a metric"""
        with self._lock:
            stats = {}
            
            if name in self.counters:
                stats['value'] = self.counters[name]
                stats['type'] = 'counter'
            
            if name in self.gauges:
                stats['value'] = self.gauges[name]
                stats['type'] = 'gauge'
            
            if name in self.histograms:
                values = self.histograms[name]
                if values:
                    stats.update({
                        'count': len(values),
                        'min': min(values),
                        'max': max(values),
                        'mean': sum(values) / len(values),
                        'type': 'histogram'
                    })
            
            if name in self.timers:
                values = self.timers[name]
                if values:
                    stats.update({
                        'count': len(values),
                        'min': min(values),
                        'max': max(values),
                        'mean': sum(values) / len(values),
                        'p95': self._percentile(values, 95),
                        'p99': self._percentile(values, 99),
                        'type': 'timer'
                    })
            
            return stats
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def clear_old_metrics(self, max_age_hours: int = 24):
        """Clear old metrics"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        with self._lock:
            self.metrics = [m for m in self.metrics if m.timestamp > cutoff_time]


class PerformanceMonitor:
    """Performance monitoring system"""
    
    def __init__(self, config: LoggingConfig):
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.snapshots: List[PerformanceSnapshot] = []
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Performance thresholds
        self.thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_percent': 90.0,
            'response_time_ms': 1000.0
        }
        
        self.logger = StructuredLogger("performance_monitor", config)
        
    def start_monitoring(self, interval_seconds: float = 5.0):
        """Start performance monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitor_thread.start()
        
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        self.logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self, interval_seconds: float):
        """Monitoring loop"""
        while self.is_monitoring:
            try:
                snapshot = self._collect_snapshot()
                self.snapshots.append(snapshot)
                
                # Keep only recent snapshots (last 1000)
                if len(self.snapshots) > 1000:
                    self.snapshots = self.snapshots[-1000:]
                
                # Check thresholds and log warnings
                self._check_thresholds(snapshot)
                
                # Update metrics
                self._update_metrics(snapshot)
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
                break
    
    def _collect_snapshot(self) -> PerformanceSnapshot:
        """Collect performance snapshot"""
        try:
            import psutil
            
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Process metrics
            process = psutil.Process()
            process_cpu = process.cpu_percent()
            process_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
            
            return PerformanceSnapshot(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / 1024 / 1024,
                memory_available_mb=memory.available / 1024 / 1024,
                disk_usage_percent=disk.percent,
                network_io=network_io,
                process_cpu_percent=process_cpu,
                process_memory_mb=process_memory,
                thread_count=process.num_threads(),
                handle_count=getattr(process, 'num_handles', None)
            )
            
        except ImportError:
            # Fallback without psutil
            return PerformanceSnapshot(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_mb=0.0,
                memory_available_mb=0.0,
                disk_usage_percent=0.0,
                network_io={},
                process_cpu_percent=0.0,
                process_memory_mb=0.0,
                thread_count=1,
                handle_count=None
            )
    
    def _check_thresholds(self, snapshot: PerformanceSnapshot):
        """Check performance thresholds"""
        context = LogContext("performance_monitor", "threshold_check")
        
        if snapshot.cpu_percent > self.thresholds['cpu_percent']:
            self.logger.warning(
                f"High CPU usage: {snapshot.cpu_percent:.1f}%",
                context=context,
                current_value=snapshot.cpu_percent,
                threshold=self.thresholds['cpu_percent']
            )
        
        if snapshot.memory_percent > self.thresholds['memory_percent']:
            self.logger.warning(
                f"High memory usage: {snapshot.memory_percent:.1f}%",
                context=context,
                current_value=snapshot.memory_percent,
                threshold=self.thresholds['memory_percent']
            )
        
        if snapshot.disk_usage_percent > self.thresholds['disk_percent']:
            self.logger.warning(
                f"High disk usage: {snapshot.disk_usage_percent:.1f}%",
                context=context,
                current_value=snapshot.disk_usage_percent,
                threshold=self.thresholds['disk_percent']
            )
    
    def _update_metrics(self, snapshot: PerformanceSnapshot):
        """Update metrics from snapshot"""
        self.metrics_collector.gauge("system.cpu_percent", snapshot.cpu_percent)
        self.metrics_collector.gauge("system.memory_percent", snapshot.memory_percent)
        self.metrics_collector.gauge("system.memory_used_mb", snapshot.memory_used_mb)
        self.metrics_collector.gauge("system.disk_usage_percent", snapshot.disk_usage_percent)
        self.metrics_collector.gauge("process.cpu_percent", snapshot.process_cpu_percent)
        self.metrics_collector.gauge("process.memory_mb", snapshot.process_memory_mb)
        self.metrics_collector.gauge("process.thread_count", snapshot.thread_count)
    
    def get_recent_snapshots(self, hours: int = 1) -> List[PerformanceSnapshot]:
        """Get recent performance snapshots"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [s for s in self.snapshots if s.timestamp > cutoff_time]
    
    def get_performance_report(self, hours: int = 1) -> Dict[str, Any]:
        """Generate performance report"""
        snapshots = self.get_recent_snapshots(hours)
        
        if not snapshots:
            return {"error": "No performance data available"}
        
        # Calculate statistics
        cpu_values = [s.cpu_percent for s in snapshots]
        memory_values = [s.memory_percent for s in snapshots]
        process_cpu_values = [s.process_cpu_percent for s in snapshots]
        process_memory_values = [s.process_memory_mb for s in snapshots]
        
        return {
            "period_hours": hours,
            "snapshot_count": len(snapshots),
            "system": {
                "cpu": {
                    "min": min(cpu_values),
                    "max": max(cpu_values),
                    "avg": sum(cpu_values) / len(cpu_values)
                },
                "memory": {
                    "min": min(memory_values),
                    "max": max(memory_values),
                    "avg": sum(memory_values) / len(memory_values)
                }
            },
            "process": {
                "cpu": {
                    "min": min(process_cpu_values),
                    "max": max(process_cpu_values),
                    "avg": sum(process_cpu_values) / len(process_cpu_values)
                },
                "memory_mb": {
                    "min": min(process_memory_values),
                    "max": max(process_memory_values),
                    "avg": sum(process_memory_values) / len(process_memory_values)
                }
            },
            "metrics": {
                name: self.metrics_collector.get_stats(name)
                for name in ["system.cpu_percent", "system.memory_percent", 
                           "process.cpu_percent", "process.memory_mb"]
            }
        }


class LoggingManager:
    """Central logging and monitoring management"""
    
    def __init__(self, config: LoggingConfig):
        self.config = config
        self.loggers: Dict[str, StructuredLogger] = {}
        self.performance_monitor = PerformanceMonitor(config)
        
        # Start performance monitoring if enabled
        if config.enable_performance_monitoring:
            self.performance_monitor.start_monitoring(
                config.performance_interval_seconds
            )
    
    def get_logger(self, name: str) -> StructuredLogger:
        """Get structured logger"""
        if name not in self.loggers:
            self.loggers[name] = StructuredLogger(name, self.config)
        return self.loggers[name]
    
    def get_performance_monitor(self) -> PerformanceMonitor:
        """Get performance monitor"""
        return self.performance_monitor
    
    def shutdown(self):
        """Shutdown logging system"""
        self.performance_monitor.stop_monitoring()
    
    def export_logs(self, output_path: str, hours: int = 24, 
                   level_filter: Optional[LogLevel] = None):
        """Export logs to file"""
        # This would aggregate logs from all loggers and export them
        # Implementation depends on how logs are stored
        pass
    
    def export_metrics(self, output_path: str, hours: int = 24):
        """Export metrics to file"""
        metrics = self.performance_monitor.metrics_collector.get_metrics()
        
        # Filter by time
        cutoff_time = datetime.now() - timedelta(hours=hours)
        filtered_metrics = [m for m in metrics if m.timestamp > cutoff_time]
        
        # Export to JSON
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump([asdict(m) for m in filtered_metrics], f, indent=2, default=str)


# Context manager for timing operations
class Timer:
    """Context manager for timing operations"""
    
    def __init__(self, name: str, metrics_collector: MetricsCollector, 
                 tags: Optional[Dict[str, str]] = None):
        self.name = name
        self.metrics_collector = metrics_collector
        self.tags = tags or {}
        self.start_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.metrics_collector.timer(self.name, duration, self.tags)


# Global logging manager instance
_logging_manager: Optional[LoggingManager] = None


def get_logging_manager() -> LoggingManager:
    """Get global logging manager"""
    global _logging_manager
    if _logging_manager is None:
        # Create default config
        config = LoggingConfig()
        _logging_manager = LoggingManager(config)
    return _logging_manager


def get_logger(name: str) -> StructuredLogger:
    """Get structured logger"""
    return get_logging_manager().get_logger(name)


def initialize_logging(config: LoggingConfig):
    """Initialize global logging manager"""
    global _logging_manager
    _logging_manager = LoggingManager(config)
    return _logging_manager


def timer(name: str, tags: Optional[Dict[str, str]] = None):
    """Decorator for timing function execution"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            metrics_collector = get_logging_manager().performance_monitor.metrics_collector
            
            with Timer(name, metrics_collector, tags):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator