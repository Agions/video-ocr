"""
Unified video processor with scene change detection and optimization
"""
import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional

import cv2
import numpy as np

from ..core.errors import VideoProcessingError
from ..models.config import ProcessingConfig
from ..models.video import VideoMetadata
from .frame_cache import FrameCache
from .scene_detection import SceneChangeDetector

logger = logging.getLogger(__name__)


@dataclass
class VideoFrame:
    """Video frame with metadata"""
    frame_id: int
    timestamp: float
    image: np.ndarray
    is_scene_change: bool = False


class UnifiedVideoProcessor:
    """
    Unified video processor with optimized frame extraction and scene change detection
    """

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.scene_detector = SceneChangeDetector(config.scene_threshold)
        self.frame_cache = FrameCache(max_size=config.cache_size)
        self.video_metadata: Optional[VideoMetadata] = None
        self._processing_stats = {
            'frames_processed': 0,
            'scenes_detected': 0,
            'cache_hits': 0,
            'processing_time': 0.0
        }
        self._thread_pool = ThreadPoolExecutor(max_workers=4)
        self._is_processing = False
        self._processing_queue = asyncio.Queue()
        self._cancellation_event = asyncio.Event()

    def get_video_info(self, video_path: str) -> VideoMetadata:
        """
        Get video metadata information

        Args:
            video_path: Path to the video file

        Returns:
            VideoMetadata: Video metadata
        """
        if not Path(video_path).exists():
            raise VideoProcessingError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise VideoProcessingError(f"Cannot open video file: {video_path}")

        try:
            return self._load_metadata(cap)
        finally:
            cap.release()

    async def extract_frames(self, video_path: str, frame_interval: float = 1.0) -> AsyncGenerator[VideoFrame, None]:
        """
        Extract frames from video with scene change detection using async processing

        Args:
            video_path: Path to the video file
            frame_interval: Interval between frames in seconds

        Yields:
            VideoFrame: Video frame with metadata
        """
        if not Path(video_path).exists():
            raise VideoProcessingError(f"Video file not found: {video_path}")

        if self._is_processing:
            raise VideoProcessingError("Video processor is already processing another video")

        self._is_processing = True
        self._cancellation_event.clear()
        
        try:
            # Start async processing task
            processing_task = asyncio.create_task(
                self._process_frames_async(video_path, frame_interval)
            )
            
            # Yield frames as they become available
            async for frame in self._yield_frames_from_queue():
                yield frame
                
            # Wait for processing to complete
            await processing_task
                
        except asyncio.CancelledError:
            self._cancellation_event.set()
            logger.info("Frame extraction cancelled by user")
            raise
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            raise
        finally:
            self._is_processing = False
            self._cancellation_event.clear()
    
    async def _process_frames_async(self, video_path: str, frame_interval: float):
        """
        Process frames asynchronously using thread pool for CPU-intensive operations
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise VideoProcessingError(f"Cannot open video file: {video_path}")

        try:
            # Load video metadata
            self.video_metadata = await asyncio.get_event_loop().run_in_executor(
                self._thread_pool, self._load_metadata_sync, cap
            )

            # Calculate frame step
            frame_step = int(frame_interval * self.video_metadata.fps)
            if frame_step < 1:
                frame_step = 1

            total_frames = self.video_metadata.frame_count
            logger.info(f"Extracting frames from video: {total_frames} total frames, interval: {frame_interval}s")

            # Process frames in batches
            batch_size = 10
            for batch_start in range(0, total_frames, frame_step * batch_size):
                if self._cancellation_event.is_set():
                    break
                    
                # Process batch of frames
                batch_tasks = []
                for i in range(batch_size):
                    frame_idx = batch_start + i * frame_step
                    if frame_idx >= total_frames:
                        break
                        
                    task = asyncio.create_task(
                        self._extract_single_frame(cap, frame_idx, self.video_metadata.fps)
                    )
                    batch_tasks.append(task)
                
                # Wait for batch to complete
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Put results in queue
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Frame extraction error: {result}")
                    elif result is not None:
                        await self._processing_queue.put(result)
                        
        except Exception as e:
            logger.error(f"Async frame processing failed: {e}")
            raise
        finally:
            cap.release()
            # Signal completion
            await self._processing_queue.put(None)
    
    async def _extract_single_frame(self, cap: cv2.VideoCapture, frame_idx: int, fps: float) -> Optional[VideoFrame]:
        """
        Extract a single frame using thread pool
        """
        def _extract_sync():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = cap.read()
            
            if not success:
                return None
                
            # Calculate timestamp
            timestamp = frame_idx / fps
            
            # Check for scene change
            is_scene_change = False
            if self.config.enable_scene_detection:
                is_scene_change = self.scene_detector.is_scene_changed(frame)
            
            # Create video frame object
            video_frame = VideoFrame(
                frame_id=frame_idx,
                timestamp=timestamp,
                image=frame,
                is_scene_change=is_scene_change
            )
            
            # Update statistics
            self._processing_stats['frames_processed'] += 1
            if is_scene_change:
                self._processing_stats['scenes_detected'] += 1
                
            return video_frame
        
        return await asyncio.get_event_loop().run_in_executor(self._thread_pool, _extract_sync)
    
    async def _yield_frames_from_queue(self) -> AsyncGenerator[VideoFrame, None]:
        """
        Yield frames from the processing queue
        """
        while True:
            try:
                frame = await asyncio.wait_for(self._processing_queue.get(), timeout=1.0)
                if frame is None:  # Completion signal
                    break
                yield frame
            except asyncio.TimeoutError:
                # Check if processing was cancelled
                if self._cancellation_event.is_set():
                    break
                continue
    
    def _load_metadata_sync(self, cap: cv2.VideoCapture) -> VideoMetadata:
        """
        Load video metadata synchronously (for thread pool execution)
        """
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        return VideoMetadata(
            file_path="",
            duration=duration,
            fps=fps,
            width=width,
            height=height,
            frame_count=frame_count,
        )


    def optimize_frame_interval(self, video_duration: float) -> float:
        """
        Calculate optimal frame interval based on video duration

        Args:
            video_duration: Video duration in seconds

        Returns:
            float: Optimal frame interval in seconds
        """
        if video_duration < 60:  # Short video (< 1 minute)
            return 0.5
        elif video_duration < 300:  # Medium video (1-5 minutes)
            return 1.0
        elif video_duration < 600:  # Long video (5-10 minutes)
            return 2.0
        else:  # Very long video (> 10 minutes)
            return 3.0

    def update_config(self, config: ProcessingConfig):
        """Update processing configuration"""
        self.config = config
        self.scene_detector = SceneChangeDetector(config.scene_threshold)
        self.frame_cache = FrameCache(max_size=config.cache_size)
    
    async def cancel_processing(self):
        """Cancel current video processing"""
        self._cancellation_event.set()
        
    def is_processing(self) -> bool:
        """Check if video processor is currently processing"""
        return self._is_processing
        
    async def __aenter__(self):
        """Async context manager entry"""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup resources"""
        self._cancellation_event.set()
        self._thread_pool.shutdown(wait=True)

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return self._processing_stats.copy()
