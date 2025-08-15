"""
Enhanced Video Processor with Performance Optimization and Security
"""
import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
from typing import Union

import cv2
import numpy as np
from pydantic import BaseModel, Field

from ..core.errors import VideoProcessingError, SecurityError
from ..core.frame_cache import EnhancedFrameCache
from ..core.memory_manager import MemoryManager
from ..models.config import ProcessingConfig
from ..models.video import VideoMetadata
from ..security.validator import validate_file_operation
from ..utils.metrics import MetricsCollector
from ..utils.performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Video processing modes"""
    SINGLE_THREADED = "single_threaded"
    MULTI_THREADED = "multi_threaded"
    ASYNC_BATCHED = "async_batched"
    GPU_ACCELERATED = "gpu_accelerated"


class FrameQuality(Enum):
    """Frame quality levels"""
    LOW = "low"      # 480p or lower
    MEDIUM = "medium"  # 720p
    HIGH = "high"     # 1080p
    ULTRA = "ultra"   # 4K or higher


@dataclass
class ProcessingOptions:
    """Video processing options"""
    mode: ProcessingMode = ProcessingMode.ASYNC_BATCHED
    quality: FrameQuality = FrameQuality.MEDIUM
    enable_scene_detection: bool = True
    enable_adaptive_sampling: bool = True
    enable_memory_optimization: bool = True
    max_concurrent_frames: int = 10
    batch_size: int = 5
    gpu_memory_limit_mb: int = 2048


@dataclass
class VideoFrame:
    """Enhanced video frame with metadata"""
    frame_id: int
    timestamp: float
    image: np.ndarray
    is_scene_change: bool = False
    quality_score: float = 0.0
    processing_time: float = 0.0
    memory_usage: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingStats:
    """Processing statistics"""
    total_frames_processed: int = 0
    total_scene_changes: int = 0
    average_processing_time: float = 0.0
    memory_usage_peak: float = 0.0
    cache_hit_rate: float = 0.0
    throughput_fps: float = 0.0
    error_count: int = 0
    success_rate: float = 100.0


class EnhancedVideoProcessor:
    """
    Enhanced video processor with performance optimization and security
    """

    def __init__(
        self, 
        config: ProcessingConfig,
        options: Optional[ProcessingOptions] = None
    ):
        self.config = config
        self.options = options or ProcessingOptions()
        
        # Initialize components
        self._initialize_components()
        
        # Processing state
        self._is_processing = False
        self._current_video: Optional[str] = None
        self._cancellation_event = asyncio.Event()
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        self.metrics_collector = MetricsCollector()
        
        # Thread pool for CPU-intensive operations
        self._thread_pool = ThreadPoolExecutor(
            max_workers=self.options.max_concurrent_frames,
            thread_name_prefix="VideoProcessor"
        )
        
        logger.info("Enhanced video processor initialized")

    def _initialize_components(self):
        """Initialize processing components"""
        try:
            # Initialize frame cache
            self.frame_cache = EnhancedFrameCache(
                max_size=self.config.cache_size or 1000,
                enable_compression=True
            )
            
            # Initialize memory manager
            self.memory_manager = MemoryManager()
            
            # Initialize scene detector
            if self.options.enable_scene_detection:
                from ..core.scene_detection import AdaptiveSceneDetector
                self.scene_detector = AdaptiveSceneDetector(
                    threshold=self.config.scene_threshold or 0.3
                )
            else:
                self.scene_detector = None
            
            logger.info("Processing components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise VideoProcessingError(f"Component initialization failed: {e}")

    async def get_video_info(self, video_path: str) -> VideoMetadata:
        """
        Get video metadata with enhanced validation and error handling
        
        Args:
            video_path: Path to the video file
            
        Returns:
            VideoMetadata: Video metadata information
            
        Raises:
            SecurityError: If file validation fails
            VideoProcessingError: If video processing fails
        """
        # Security validation
        if not validate_file_operation(video_path, 'video'):
            raise SecurityError(f"Security validation failed for video: {video_path}")
        
        try:
            # Use thread pool for file operations
            return await asyncio.get_event_loop().run_in_executor(
                self._thread_pool,
                self._load_video_metadata,
                video_path
            )
            
        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            raise VideoProcessingError(f"Failed to get video info: {e}")

    def _load_video_metadata(self, video_path: str) -> VideoMetadata:
        """Load video metadata synchronously"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise VideoProcessingError(f"Cannot open video file: {video_path}")
            
            try:
                # Get video properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                
                # Get additional metadata
                codec = int(cap.get(cv2.CAP_PROP_FOURCC))
                bitrate = int(cap.get(cv2.CAP_PROP_BITRATE))
                
                return VideoMetadata(
                    file_path=video_path,
                    duration=duration,
                    fps=fps,
                    width=width,
                    height=height,
                    frame_count=frame_count,
                    codec=codec,
                    bitrate=bitrate,
                    quality_level=self._determine_quality_level(width, height)
                )
                
            finally:
                cap.release()
                
        except Exception as e:
            logger.error(f"Failed to load video metadata: {e}")
            raise VideoProcessingError(f"Failed to load video metadata: {e}")

    def _determine_quality_level(self, width: int, height: int) -> FrameQuality:
        """Determine video quality level based on resolution"""
        total_pixels = width * height
        
        if total_pixels <= 640 * 480:
            return FrameQuality.LOW
        elif total_pixels <= 1280 * 720:
            return FrameQuality.MEDIUM
        elif total_pixels <= 1920 * 1080:
            return FrameQuality.HIGH
        else:
            return FrameQuality.ULTRA

    async def extract_frames(
        self, 
        video_path: str, 
        frame_interval: float = 1.0
    ) -> AsyncGenerator[VideoFrame, None]:
        """
        Extract frames from video with enhanced performance and security
        
        Args:
            video_path: Path to the video file
            frame_interval: Interval between frames in seconds
            
        Yields:
            VideoFrame: Enhanced video frame with metadata
            
        Raises:
            SecurityError: If file validation fails
            VideoProcessingError: If video processing fails
        """
        # Security validation
        if not validate_file_operation(video_path, 'video'):
            raise SecurityError(f"Security validation failed for video: {video_path}")
        
        if self._is_processing:
            raise VideoProcessingError("Video processor is already processing another video")
        
        self._is_processing = True
        self._current_video = video_path
        self._cancellation_event.clear()
        
        try:
            # Start processing based on mode
            if self.options.mode == ProcessingMode.ASYNC_BATCHED:
                async for frame in self._extract_frames_async_batched(
                    video_path, frame_interval
                ):
                    yield frame
            elif self.options.mode == ProcessingMode.MULTI_THREADED:
                async for frame in self._extract_frames_multi_threaded(
                    video_path, frame_interval
                ):
                    yield frame
            else:
                async for frame in self._extract_frames_single_threaded(
                    video_path, frame_interval
                ):
                    yield frame
                    
        except asyncio.CancelledError:
            logger.info("Frame extraction cancelled")
            raise
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            raise VideoProcessingError(f"Frame extraction failed: {e}")
        finally:
            self._is_processing = False
            self._current_video = None
            self._cancellation_event.clear()

    async def _extract_frames_async_batched(
        self, 
        video_path: str, 
        frame_interval: float
    ) -> AsyncGenerator[VideoFrame, None]:
        """Extract frames using async batched processing"""
        try:
            # Get video info
            video_info = await self.get_video_info(video_path)
            
            # Calculate frame step
            frame_step = int(frame_interval * video_info.fps)
            if frame_step < 1:
                frame_step = 1
            
            # Calculate adaptive frame interval if enabled
            if self.options.enable_adaptive_sampling:
                frame_interval = self._calculate_adaptive_interval(
                    video_info.duration, video_info.frame_count
                )
                frame_step = int(frame_interval * video_info.fps)
            
            logger.info(f"Starting async batched frame extraction: {video_info.frame_count} frames, interval: {frame_interval}s")
            
            # Process frames in batches
            processing_queue = asyncio.Queue()
            
            # Start producer task
            producer_task = asyncio.create_task(
                self._produce_frames_batched(
                    video_path, frame_step, video_info, processing_queue
                )
            )
            
            # Start consumer task
            consumer_task = asyncio.create_task(
                self._consume_frames_batched(processing_queue)
            )
            
            # Yield frames as they become available
            async for frame in self._yield_frames_from_queue(processing_queue):
                yield frame
            
            # Wait for tasks to complete
            await producer_task
            await consumer_task
            
        except Exception as e:
            logger.error(f"Async batched extraction failed: {e}")
            raise

    async def _produce_frames_batched(
        self,
        video_path: str,
        frame_step: int,
        video_info: VideoMetadata,
        output_queue: asyncio.Queue
    ):
        """Produce frames in batches"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise VideoProcessingError(f"Cannot open video file: {video_path}")
            
            try:
                batch_size = self.options.batch_size
                total_frames = video_info.frame_count
                
                for batch_start in range(0, total_frames, frame_step * batch_size):
                    if self._cancellation_event.is_set():
                        break
                    
                    # Create batch tasks
                    batch_tasks = []
                    for i in range(batch_size):
                        frame_idx = batch_start + i * frame_step
                        if frame_idx >= total_frames:
                            break
                        
                        task = asyncio.create_task(
                            self._extract_single_frame_enhanced(
                                cap, frame_idx, video_info.fps
                            )
                        )
                        batch_tasks.append(task)
                    
                    # Wait for batch to complete
                    batch_results = await asyncio.gather(
                        *batch_tasks, 
                        return_exceptions=True
                    )
                    
                    # Put results in queue
                    for result in batch_results:
                        if isinstance(result, Exception):
                            logger.error(f"Frame extraction error: {result}")
                        elif result is not None:
                            await output_queue.put(result)
                
                # Signal completion
                await output_queue.put(None)
                
            finally:
                cap.release()
                
        except Exception as e:
            logger.error(f"Frame production failed: {e}")
            await output_queue.put(None)

    async def _consume_frames_batched(self, input_queue: asyncio.Queue):
        """Consume and process frames from queue"""
        try:
            while True:
                frame = await input_queue.get()
                if frame is None:  # Completion signal
                    break
                
                # Apply post-processing
                processed_frame = await self._post_process_frame(frame)
                
                # Update metrics
                self.metrics_collector.record_frame_processing_time(
                    processed_frame.processing_time
                )
                
        except Exception as e:
            logger.error(f"Frame consumption failed: {e}")

    async def _extract_single_frame_enhanced(
        self,
        cap: cv2.VideoCapture,
        frame_idx: int,
        fps: float
    ) -> Optional[VideoFrame]:
        """Extract a single frame with enhanced processing"""
        try:
            start_time = time.time()
            
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = cap.read()
            
            if not success:
                return None
            
            # Calculate timestamp
            timestamp = frame_idx / fps
            
            # Apply scene detection
            is_scene_change = False
            if self.scene_detector:
                is_scene_change = await asyncio.get_event_loop().run_in_executor(
                    self._thread_pool,
                    self.scene_detector.is_scene_changed,
                    frame
                )
            
            # Calculate quality score
            quality_score = self._calculate_frame_quality(frame)
            
            # Apply quality-based resizing
            processed_frame = await self._apply_quality_processing(frame)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create video frame object
            video_frame = VideoFrame(
                frame_id=frame_idx,
                timestamp=timestamp,
                image=processed_frame,
                is_scene_change=is_scene_change,
                quality_score=quality_score,
                processing_time=processing_time,
                memory_usage=processed_frame.nbytes,
                metadata={
                    'original_size': frame.shape,
                    'processed_size': processed_frame.shape,
                    'compression_ratio': frame.nbytes / processed_frame.nbytes
                }
            )
            
            return video_frame
            
        except Exception as e:
            logger.error(f"Single frame extraction failed: {e}")
            return None

    def _calculate_frame_quality(self, frame: np.ndarray) -> float:
        """Calculate frame quality score"""
        try:
            # Simple quality metrics
            blur_score = cv2.Laplacian(frame, cv2.CV_64F).var()
            brightness = np.mean(frame)
            contrast = np.std(frame)
            
            # Normalize scores
            blur_score = min(1.0, blur_score / 1000.0)
            brightness_score = 1.0 - abs(brightness - 128) / 128.0
            contrast_score = min(1.0, contrast / 64.0)
            
            # Combined quality score
            quality_score = (blur_score + brightness_score + contrast_score) / 3.0
            
            return quality_score
            
        except Exception as e:
            logger.error(f"Quality calculation failed: {e}")
            return 0.5

    async def _apply_quality_processing(self, frame: np.ndarray) -> np.ndarray:
        """Apply quality-based processing to frame"""
        try:
            processed_frame = frame.copy()
            
            # Apply quality-based resizing
            if self.options.quality == FrameQuality.LOW:
                # Resize to 480p
                height, width = processed_frame.shape[:2]
                if height > 480 or width > 854:
                    processed_frame = cv2.resize(
                        processed_frame, (854, 480)
                    )
            elif self.options.quality == FrameQuality.MEDIUM:
                # Resize to 720p
                height, width = processed_frame.shape[:2]
                if height > 720 or width > 1280:
                    processed_frame = cv2.resize(
                        processed_frame, (1280, 720)
                    )
            
            # Apply noise reduction if needed
            if self.options.quality == FrameQuality.LOW:
                processed_frame = cv2.fastNlMeansDenoisingColored(
                    processed_frame, None, 10, 10, 7, 21
                )
            
            return processed_frame
            
        except Exception as e:
            logger.error(f"Quality processing failed: {e}")
            return frame

    async def _post_process_frame(self, frame: VideoFrame) -> VideoFrame:
        """Post-process frame with additional optimizations"""
        try:
            # Cache frame if quality is high enough
            if frame.quality_score > 0.7:
                self.frame_cache.add_frame(
                    frame.frame_id, frame.image, frame.timestamp
                )
            
            # Update performance metrics
            self.performance_monitor.record_frame_processed(
                frame.processing_time, frame.memory_usage
            )
            
            return frame
            
        except Exception as e:
            logger.error(f"Frame post-processing failed: {e}")
            return frame

    async def _extract_frames_multi_threaded(
        self, 
        video_path: str, 
        frame_interval: float
    ) -> AsyncGenerator[VideoFrame, None]:
        """Extract frames using multi-threaded processing"""
        try:
            video_info = await self.get_video_info(video_path)
            frame_step = int(frame_interval * video_info.fps)
            
            logger.info(f"Starting multi-threaded frame extraction: {video_info.frame_count} frames")
            
            # Create video capture
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise VideoProcessingError(f"Cannot open video file: {video_path}")
            
            try:
                # Process frames in parallel
                frame_indices = range(0, video_info.frame_count, frame_step)
                
                # Create tasks for parallel processing
                tasks = []
                for frame_idx in frame_indices:
                    if self._cancellation_event.is_set():
                        break
                    
                    task = asyncio.create_task(
                        self._extract_single_frame_enhanced(
                            cap, frame_idx, video_info.fps
                        )
                    )
                    tasks.append(task)
                
                # Process results as they complete
                for future in asyncio.as_completed(tasks):
                    if self._cancellation_event.is_set():
                        break
                    
                    frame = await future
                    if frame is not None:
                        yield frame
                        
            finally:
                cap.release()
                
        except Exception as e:
            logger.error(f"Multi-threaded extraction failed: {e}")
            raise

    async def _extract_frames_single_threaded(
        self, 
        video_path: str, 
        frame_interval: float
    ) -> AsyncGenerator[VideoFrame, None]:
        """Extract frames using single-threaded processing"""
        try:
            video_info = await self.get_video_info(video_path)
            frame_step = int(frame_interval * video_info.fps)
            
            logger.info(f"Starting single-threaded frame extraction: {video_info.frame_count} frames")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise VideoProcessingError(f"Cannot open video file: {video_path}")
            
            try:
                for frame_idx in range(0, video_info.frame_count, frame_step):
                    if self._cancellation_event.is_set():
                        break
                    
                    frame = await self._extract_single_frame_enhanced(
                        cap, frame_idx, video_info.fps
                    )
                    
                    if frame is not None:
                        yield frame
                        
            finally:
                cap.release()
                
        except Exception as e:
            logger.error(f"Single-threaded extraction failed: {e}")
            raise

    async def _yield_frames_from_queue(
        self, 
        queue: asyncio.Queue
    ) -> AsyncGenerator[VideoFrame, None]:
        """Yield frames from processing queue"""
        while True:
            try:
                frame = await asyncio.wait_for(queue.get(), timeout=1.0)
                if frame is None:  # Completion signal
                    break
                yield frame
            except asyncio.TimeoutError:
                if self._cancellation_event.is_set():
                    break
                continue

    def _calculate_adaptive_interval(
        self, 
        duration: float, 
        frame_count: int
    ) -> float:
        """Calculate adaptive frame interval based on video characteristics"""
        if duration < 60:  # Short video
            return 0.5
        elif duration < 300:  # Medium video
            return 1.0
        elif duration < 600:  # Long video
            return 2.0
        else:  # Very long video
            return 3.0

    def optimize_frame_interval(self, video_duration: float) -> float:
        """Calculate optimal frame interval based on video duration"""
        return self._calculate_adaptive_interval(video_duration, 0)

    def update_config(self, config: ProcessingConfig):
        """Update processing configuration"""
        self.config = config
        
        # Reinitialize components if needed
        if self.options.enable_scene_detection:
            from ..core.scene_detection import AdaptiveSceneDetector
            self.scene_detector = AdaptiveSceneDetector(
                threshold=self.config.scene_threshold or 0.3
            )

    async def cancel_processing(self):
        """Cancel current video processing"""
        self._cancellation_event.set()
        logger.info("Video processing cancelled")

    def is_processing(self) -> bool:
        """Check if video processor is currently processing"""
        return self._is_processing

    async def get_processing_stats(self) -> ProcessingStats:
        """Get current processing statistics"""
        return ProcessingStats(
            total_frames_processed=self.metrics_collector.get_total_frames_processed(),
            total_scene_changes=self.metrics_collector.get_total_scene_changes(),
            average_processing_time=self.metrics_collector.get_average_processing_time(),
            memory_usage_peak=self.performance_monitor.get_peak_memory_usage(),
            cache_hit_rate=self.frame_cache.get_hit_rate(),
            throughput_fps=self.performance_monitor.get_throughput_fps(),
            error_count=self.metrics_collector.get_error_count(),
            success_rate=self.metrics_collector.get_success_rate()
        )

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        return {
            'processing_stats': await self.get_processing_stats(),
            'performance_monitor': self.performance_monitor.get_metrics(),
            'memory_info': await self.memory_manager.get_memory_info(),
            'cache_info': self.frame_cache.get_cache_info()
        }

    @asynccontextmanager
    async def processing_context(self):
        """Async context manager for processing"""
        try:
            yield self
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Cancel any ongoing processing
            await self.cancel_processing()
            
            # Shutdown thread pool
            self._thread_pool.shutdown(wait=True)
            
            # Cleanup cache
            self.frame_cache.clear()
            
            logger.info("Enhanced video processor cleaned up")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    def __del__(self):
        """Destructor"""
        if hasattr(self, '_thread_pool'):
            self._thread_pool.shutdown(wait=False)