"""
Enhanced Core Engine with Security, Performance, and Reliability Improvements
"""
import asyncio
import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, AsyncGenerator, Callable
from typing import Union

import numpy as np
from pydantic import BaseModel, Field

from ..core.config_manager import config_manager
from ..core.errors import (
    VideoProcessingError, OCRError, SecurityError, 
    ProcessingError, ConfigurationError
)
from ..core.memory_manager import MemoryManager
from ..core.ocr_engine import OCREngineFactory
from ..core.roi_manager import ROIManager
from ..core.subtitle_processor import SubtitleProcessor, ProcessingContext
from ..core.video_processor import UnifiedVideoProcessor
from ..models.config import ProcessingConfig
from ..models.subtitle import SubtitleItem
from ..security.validator import validate_file_operation, SecurityPolicy
from ..security.secure_storage import SecureStorage, SecureCache
from ..services.batch_service import BatchProcessingService
from ..utils.metrics import MetricsCollector
from ..utils.health_monitor import HealthMonitor

logger = logging.getLogger(__name__)


class ProcessingState(Enum):
    """Processing state enumeration"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    EXTRACTING_FRAMES = "extracting_frames"
    PROCESSING_OCR = "processing_ocr"
    GENERATING_SUBTITLES = "generating_subtitles"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProcessingPriority(Enum):
    """Processing priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ProcessingTask:
    """Processing task definition"""
    id: str
    video_path: str
    config: ProcessingConfig
    priority: ProcessingPriority = ProcessingPriority.NORMAL
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    state: ProcessingState = ProcessingState.IDLE
    progress: float = 0.0
    error: Optional[str] = None
    result: Optional[List[SubtitleItem]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingMetrics:
    """Processing metrics collection"""
    total_frames_processed: int = 0
    total_ocr_operations: int = 0
    average_processing_time: float = 0.0
    memory_usage_peak: float = 0.0
    cache_hit_rate: float = 0.0
    error_count: int = 0
    success_rate: float = 0.0
    throughput_fps: float = 0.0


class EnhancedProcessingEngine:
    """
    Enhanced processing engine with improved security, performance, and reliability
    """

    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self._setup_components()
        self._setup_security()
        self._setup_monitoring()
        
        # Processing state
        self._current_task: Optional[ProcessingTask] = None
        self._processing_queue: asyncio.Queue[ProcessingTask] = asyncio.Queue()
        self._is_running = False
        self._processing_lock = asyncio.Lock()
        
        # Performance optimization
        self._thread_pool = ThreadPoolExecutor(
            max_workers=self.config.max_workers or 4,
            thread_name_prefix="VisionSub-Worker"
        )
        self._memory_manager = MemoryManager()
        self._batch_service = BatchProcessingService()
        
        # Event handling
        self._event_handlers: Dict[str, List[Callable]] = {}
        
        logger.info("Enhanced processing engine initialized")

    def _setup_components(self):
        """Setup core processing components"""
        try:
            # Initialize video processor with enhanced settings
            self.video_processor = UnifiedVideoProcessor(self.config)
            
            # Initialize OCR engine with security checks
            ocr_config = self.config.ocr_config.model_dump()
            self._validate_ocr_config(ocr_config)
            self.ocr_service = OCREngineFactory.create_engine(
                self.config.ocr_config.engine,
                ocr_config
            )
            
            # Initialize ROI manager
            self.roi_manager = ROIManager()
            
            # Initialize subtitle processor
            self.subtitle_processor = SubtitleProcessor(self.config)
            
            # Initialize secure storage and cache
            self.secure_storage = SecureStorage()
            self.secure_cache = SecureCache()
            
            logger.info("Core components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup components: {e}")
            raise ConfigurationError(f"Component initialization failed: {e}")

    def _setup_security(self):
        """Setup security components and policies"""
        try:
            # Configure security policy
            self.security_policy = SecurityPolicy(
                max_file_size=self.config.max_file_size or 1024 * 1024 * 1024,  # 1GB
                security_level=SecurityPolicy.SecurityLevel.HIGH,
                sanitize_paths=True,
                validate_mime_types=True
            )
            
            # Initialize security validator
            from ..security.validator import get_security_validator
            self.security_validator = get_security_validator(self.security_policy)
            
            logger.info("Security components initialized")
            
        except Exception as e:
            logger.error(f"Failed to setup security: {e}")
            raise ConfigurationError(f"Security initialization failed: {e}")

    def _setup_monitoring(self):
        """Setup monitoring and metrics collection"""
        try:
            # Initialize metrics collector
            self.metrics_collector = MetricsCollector()
            
            # Initialize health monitor
            self.health_monitor = HealthMonitor()
            
            # Register health checks
            self.health_monitor.register_check(
                "ocr_engine", self._check_ocr_health
            )
            self.health_monitor.register_check(
                "memory", self._check_memory_health
            )
            self.health_monitor.register_check(
                "storage", self._check_storage_health
            )
            
            logger.info("Monitoring components initialized")
            
        except Exception as e:
            logger.error(f"Failed to setup monitoring: {e}")
            raise ConfigurationError(f"Monitoring initialization failed: {e}")

    def _validate_ocr_config(self, ocr_config: Dict[str, Any]):
        """Validate OCR configuration for security"""
        try:
            # Check for dangerous configurations
            if 'model_path' in ocr_config:
                model_path = Path(ocr_config['model_path'])
                if not self.security_validator.validate_file_path(model_path):
                    raise SecurityError(f"Invalid OCR model path: {model_path}")
            
            # Validate numeric ranges
            for key, value in ocr_config.items():
                if not self.security_validator.validate_config_value(key, value):
                    raise SecurityError(f"Invalid OCR configuration: {key}={value}")
                    
        except Exception as e:
            logger.error(f"OCR config validation failed: {e}")
            raise SecurityError(f"OCR configuration validation failed: {e}")

    async def _check_ocr_health(self) -> bool:
        """Check OCR engine health"""
        try:
            if not self.ocr_service.is_initialized():
                return False
            
            # Test with a simple image
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
            result = await self.ocr_service.process_image(test_image)
            return result is not None
            
        except Exception as e:
            logger.error(f"OCR health check failed: {e}")
            return False

    async def _check_memory_health(self) -> bool:
        """Check memory health"""
        try:
            return await self._memory_manager.check_memory_health()
        except Exception as e:
            logger.error(f"Memory health check failed: {e}")
            return False

    async def _check_storage_health(self) -> bool:
        """Check storage health"""
        try:
            # Check secure storage
            storage_info = self.secure_storage.get_config_info()
            return storage_info.get('key_exists', False)
        except Exception as e:
            logger.error(f"Storage health check failed: {e}")
            return False

    async def process_video(
        self, 
        video_path: str, 
        config: Optional[ProcessingConfig] = None,
        priority: ProcessingPriority = ProcessingPriority.NORMAL
    ) -> List[SubtitleItem]:
        """
        Process a video file with enhanced security and performance
        
        Args:
            video_path: Path to the video file
            config: Optional processing configuration
            priority: Processing priority level
            
        Returns:
            List of extracted subtitle items
            
        Raises:
            SecurityError: If security validation fails
            ProcessingError: If processing fails
        """
        # Validate input
        if not await self._validate_video_input(video_path):
            raise SecurityError(f"Security validation failed for video: {video_path}")
        
        # Create processing task
        task_id = f"task_{int(time.time() * 1000)}"
        task = ProcessingTask(
            id=task_id,
            video_path=video_path,
            config=config or self.config,
            priority=priority
        )
        
        # Emit start event
        await self._emit_event("processing_started", task)
        
        try:
            # Process the video
            result = await self._process_task(task)
            
            # Update task state
            task.state = ProcessingState.COMPLETED
            task.result = result
            task.completed_at = time.time()
            
            # Emit completion event
            await self._emit_event("processing_completed", task)
            
            return result
            
        except Exception as e:
            # Update task state
            task.state = ProcessingState.FAILED
            task.error = str(e)
            task.completed_at = time.time()
            
            # Emit error event
            await self._emit_event("processing_failed", task)
            
            logger.error(f"Video processing failed: {e}")
            raise ProcessingError(f"Video processing failed: {e}")

    async def _validate_video_input(self, video_path: str) -> bool:
        """Validate video input with security checks"""
        try:
            # Check file path security
            if not validate_file_operation(video_path, 'video'):
                logger.warning(f"Security validation failed for video: {video_path}")
                return False
            
            # Check file existence and accessibility
            video_file = Path(video_path)
            if not video_file.exists():
                logger.warning(f"Video file does not exist: {video_path}")
                return False
            
            # Check file size
            file_size = video_file.stat().st_size
            if file_size > self.security_policy.max_file_size:
                logger.warning(f"Video file too large: {file_size} bytes")
                return False
            
            # Check file permissions
            if not video_file.is_file():
                logger.warning(f"Path is not a file: {video_path}")
                return False
            
            logger.debug(f"Video input validation passed: {video_path}")
            return True
            
        except Exception as e:
            logger.error(f"Video input validation error: {e}")
            return False

    async def _process_task(self, task: ProcessingTask) -> List[SubtitleItem]:
        """Process a single task with enhanced error handling"""
        async with self._processing_lock:
            if self._current_task is not None:
                raise ProcessingError("Another task is already processing")
            
            self._current_task = task
            task.state = ProcessingState.INITIALIZING
            task.started_at = time.time()
        
        try:
            # Start processing
            await self._emit_event("task_started", task)
            
            # Get video information
            task.state = ProcessingState.INITIALIZING
            video_info = await self._get_video_info(task.video_path)
            task.metadata['video_info'] = video_info.dict()
            
            # Create processing context
            context = ProcessingContext(
                video_duration=video_info.duration,
                fps=video_info.fps,
                frame_count=video_info.frame_count,
                video_width=video_info.width,
                video_height=video_info.height
            )
            
            # Extract and process frames
            task.state = ProcessingState.EXTRACTING_FRAMES
            ocr_results = await self._extract_and_process_frames(task, context)
            
            # Generate subtitles
            task.state = ProcessingState.GENERATING_SUBTITLES
            subtitles = await self.subtitle_processor.process_ocr_results(
                ocr_results, context
            )
            
            # Finalize processing
            task.state = ProcessingState.FINALIZING
            await self._finalize_processing(task, subtitles)
            
            logger.info(f"Task {task.id} completed successfully")
            return subtitles
            
        except asyncio.CancelledError:
            task.state = ProcessingState.CANCELLED
            logger.info(f"Task {task.id} cancelled")
            raise
        except Exception as e:
            task.state = ProcessingState.FAILED
            task.error = str(e)
            logger.error(f"Task {task.id} failed: {e}")
            raise
        finally:
            self._current_task = None

    async def _get_video_info(self, video_path: str):
        """Get video information with error handling"""
        try:
            return await asyncio.get_event_loop().run_in_executor(
                self._thread_pool,
                self.video_processor.get_video_info,
                video_path
            )
        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            raise VideoProcessingError(f"Failed to get video info: {e}")

    async def _extract_and_process_frames(
        self, 
        task: ProcessingTask, 
        context: ProcessingContext
    ) -> List[tuple]:
        """Extract and process frames with batching and caching"""
        ocr_results = []
        frame_interval = task.config.frame_interval
        
        # Calculate optimal batch size
        batch_size = self._calculate_optimal_batch_size(context.frame_count)
        
        try:
            # Process frames in batches
            async for frame_data in self.video_processor.extract_frames(
                task.video_path, frame_interval
            ):
                if task.state == ProcessingState.CANCELLED:
                    break
                
                frame, timestamp = frame_data
                
                # Apply ROI processing
                processed_frame = await self._apply_roi_processing(frame)
                
                # Process OCR with caching
                ocr_result = await self._process_ocr_with_cache(
                    processed_frame, timestamp
                )
                
                ocr_results.append((ocr_result, timestamp))
                
                # Update progress
                task.progress = min(100.0, (len(ocr_results) / context.frame_count) * 100)
                await self._emit_event("progress_update", task)
                
                # Memory management
                await self._memory_manager.check_memory_usage()
                
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            raise VideoProcessingError(f"Frame extraction failed: {e}")
        
        return ocr_results

    def _calculate_optimal_batch_size(self, frame_count: int) -> int:
        """Calculate optimal batch size based on system resources"""
        # Simple heuristic based on frame count and available workers
        if frame_count < 100:
            return 5
        elif frame_count < 1000:
            return 10
        else:
            return 20

    async def _apply_roi_processing(self, frame: np.ndarray) -> np.ndarray:
        """Apply ROI processing to frame"""
        try:
            return await asyncio.get_event_loop().run_in_executor(
                self._thread_pool,
                self.roi_manager.apply_roi_to_frame,
                frame
            )
        except Exception as e:
            logger.error(f"ROI processing failed: {e}")
            # Return original frame if ROI processing fails
            return frame

    async def _process_ocr_with_cache(
        self, 
        frame: np.ndarray, 
        timestamp: float
    ):
        """Process OCR with caching support"""
        try:
            # Generate cache key
            cache_key = f"ocr_{hash(frame.tobytes())}_{timestamp}"
            
            # Check cache first
            cached_result = self.secure_cache.retrieve_data(cache_key)
            if cached_result:
                logger.debug("OCR cache hit")
                return cached_result
            
            # Process OCR
            ocr_result = await self.ocr_service.process_image(frame)
            
            # Cache result (with TTL)
            self.secure_cache.store_data(cache_key, ocr_result, ttl_hours=1)
            
            return ocr_result
            
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            raise OCRError(f"OCR processing failed: {e}")

    async def _finalize_processing(
        self, 
        task: ProcessingTask, 
        subtitles: List[SubtitleItem]
    ):
        """Finalize processing with cleanup and metrics collection"""
        try:
            # Collect metrics
            metrics = ProcessingMetrics(
                total_frames_processed=len(subtitles),
                total_ocr_operations=len(subtitles),
                success_rate=100.0 if subtitles else 0.0,
                throughput_fps=len(subtitles) / max(1, task.completed_at - task.started_at)
            )
            
            task.metadata['metrics'] = metrics.__dict__
            
            # Store results securely
            await self._store_processing_results(task, subtitles)
            
            # Update metrics collector
            self.metrics_collector.record_processing_time(
                task.completed_at - task.started_at
            )
            
            logger.debug(f"Processing finalized for task {task.id}")
            
        except Exception as e:
            logger.error(f"Processing finalization failed: {e}")
            # Don't raise here as processing is already complete

    async def _store_processing_results(
        self, 
        task: ProcessingTask, 
        subtitles: List[SubtitleItem]
    ):
        """Store processing results securely"""
        try:
            # Convert subtitles to serializable format
            subtitles_data = [subtitle.dict() for subtitle in subtitles]
            
            # Store in secure storage
            storage_key = f"processing_result_{task.id}"
            self.secure_storage.save_config(
                {
                    'task_id': task.id,
                    'video_path': task.video_path,
                    'subtitles': subtitles_data,
                    'metadata': task.metadata,
                    'timestamp': time.time()
                }
            )
            
            logger.debug(f"Processing results stored for task {task.id}")
            
        except Exception as e:
            logger.error(f"Failed to store processing results: {e}")

    async def _emit_event(self, event_type: str, data: Any):
        """Emit event to registered handlers"""
        if event_type in self._event_handlers:
            for handler in self._event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                except Exception as e:
                    logger.error(f"Event handler error: {e}")

    def add_event_handler(self, event_type: str, handler: Callable):
        """Add event handler"""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    def remove_event_handler(self, event_type: str, handler: Callable):
        """Remove event handler"""
        if event_type in self._event_handlers:
            self._event_handlers[event_type].remove(handler)

    async def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status"""
        if self._current_task:
            return {
                'is_processing': True,
                'current_task': {
                    'id': self._current_task.id,
                    'video_path': self._current_task.video_path,
                    'state': self._current_task.state.value,
                    'progress': self._current_task.progress,
                    'started_at': self._current_task.started_at,
                    'priority': self._current_task.priority.value
                },
                'queue_size': self._processing_queue.qsize(),
                'health_status': await self.health_monitor.get_health_status()
            }
        else:
            return {
                'is_processing': False,
                'queue_size': self._processing_queue.qsize(),
                'health_status': await self.health_monitor.get_health_status()
            }

    async def get_metrics(self) -> Dict[str, Any]:
        """Get processing metrics"""
        return {
            'metrics': self.metrics_collector.get_metrics(),
            'health': await self.health_monitor.get_health_status(),
            'memory': await self._memory_manager.get_memory_info(),
            'security': self.security_validator.get_security_info()
        }

    async def cancel_processing(self):
        """Cancel current processing"""
        if self._current_task:
            self._current_task.state = ProcessingState.CANCELLED
            await self.video_processor.cancel_processing()
            logger.info("Processing cancelled")

    async def start_processing_queue(self):
        """Start the processing queue"""
        if self._is_running:
            return
        
        self._is_running = True
        asyncio.create_task(self._process_queue())
        logger.info("Processing queue started")

    async def stop_processing_queue(self):
        """Stop the processing queue"""
        self._is_running = False
        await self.cancel_processing()
        logger.info("Processing queue stopped")

    async def _process_queue(self):
        """Process items from the queue"""
        while self._is_running:
            try:
                # Get next task with timeout
                task = await asyncio.wait_for(
                    self._processing_queue.get(), 
                    timeout=1.0
                )
                
                # Process the task
                await self._process_task(task)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Queue processing error: {e}")
                continue

    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Stop processing
            await self.stop_processing_queue()
            
            # Cleanup thread pool
            self._thread_pool.shutdown(wait=True)
            
            # Cleanup OCR engines
            OCREngineFactory.cleanup_all()
            
            # Cleanup monitoring
            await self.health_monitor.cleanup()
            
            logger.info("Enhanced processing engine cleaned up")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    @asynccontextmanager
    async def processing_context(self):
        """Async context manager for processing"""
        try:
            yield self
        finally:
            await self.cleanup()

    def __del__(self):
        """Destructor"""
        if hasattr(self, '_thread_pool'):
            self._thread_pool.shutdown(wait=False)