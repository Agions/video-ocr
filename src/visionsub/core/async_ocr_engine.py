"""
Asynchronous OCR processing engine with concurrent processing
"""
import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import numpy as np

from ..core.errors import OCRError
from ..models.config import OcrConfig
from ..models.ocr import OCRResult
from .ocr_engine import OCREngine
from .text_processor import TextProcessor

logger = logging.getLogger(__name__)


@dataclass
class OCRJob:
    """OCR processing job with metadata"""
    job_id: str
    frame_id: int
    timestamp: float
    image: np.ndarray
    roi_coords: Optional[Tuple[int, int, int, int]] = None
    priority: int = 0


@dataclass
class OCRJobResult:
    """Result of OCR job processing"""
    job_id: str
    result: OCRResult
    processing_time: float
    success: bool
    error_message: Optional[str] = None


class AsyncOCREngine:
    """
    Asynchronous OCR processing engine with concurrent processing and job queuing
    """

    def __init__(self, config: OcrConfig, max_workers: int = 4):
        self.config = config
        self.ocr_engine = OCREngine(config)
        self.text_processor = TextProcessor(config)
        
        # Thread pool for CPU-intensive OCR operations
        self._thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # Job processing queue
        self._job_queue = asyncio.PriorityQueue()
        self._result_queue = asyncio.Queue()
        
        # Processing state
        self._is_processing = False
        self._processing_stats = {
            'jobs_completed': 0,
            'jobs_failed': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'concurrent_jobs': 0,
            'jobs_per_second': 0.0
        }
        
        # Cancellation and shutdown
        self._cancellation_event = asyncio.Event()
        self._shutdown_event = asyncio.Event()
        
        # Performance monitoring
        self._processing_times: List[float] = []
        self._max_processing_times = 1000  # Keep last 1000 processing times

    async def process_frame(
        self, 
        frame_id: int, 
        timestamp: float, 
        image: np.ndarray,
        roi_coords: Optional[Tuple[int, int, int, int]] = None,
        priority: int = 0
    ) -> OCRJobResult:
        """
        Process a single frame asynchronously
        
        Args:
            frame_id: Frame identifier
            timestamp: Frame timestamp
            image: Frame image data
            roi_coords: Region of interest coordinates (x, y, width, height)
            priority: Processing priority (higher = higher priority)
            
        Returns:
            OCRJobResult: Processing result
        """
        job_id = f"frame_{frame_id}_{int(time.time() * 1000)}"
        job = OCRJob(
            job_id=job_id,
            frame_id=frame_id,
            timestamp=timestamp,
            image=image,
            roi_coords=roi_coords,
            priority=priority
        )
        
        return await self._process_single_job(job)

    async def process_frames_batch(
        self, 
        frames: List[Tuple[int, float, np.ndarray, Optional[Tuple[int, int, int, int]]]],
        max_concurrent: int = 4
    ) -> List[OCRJobResult]:
        """
        Process multiple frames concurrently
        
        Args:
            frames: List of (frame_id, timestamp, image, roi_coords) tuples
            max_concurrent: Maximum concurrent processing jobs
            
        Returns:
            List[OCRJobResult]: Processing results
        """
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def _process_with_semaphore(frame_data):
            async with semaphore:
                frame_id, timestamp, image, roi_coords = frame_data
                return await self.process_frame(frame_id, timestamp, image, roi_coords)
        
        # Process all frames concurrently with limited concurrency
        tasks = [
            _process_with_semaphore(frame_data) 
            for frame_data in frames
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return successful results
        successful_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Frame processing failed: {result}")
            else:
                successful_results.append(result)
                
        return successful_results

    async def start_background_processing(self):
        """Start background processing of queued jobs"""
        if self._is_processing:
            return
            
        self._is_processing = True
        self._cancellation_event.clear()
        
        # Start worker tasks
        num_workers = min(4, self._thread_pool._max_workers)
        self._worker_tasks = [
            asyncio.create_task(self._worker_loop(f"worker_{i}"))
            for i in range(num_workers)
        ]
        
        # Start stats monitoring
        self._stats_task = asyncio.create_task(self._update_stats_periodically())
        
        logger.info(f"Started background OCR processing with {num_workers} workers")

    async def stop_background_processing(self):
        """Stop background processing and wait for completion"""
        if not self._is_processing:
            return
            
        self._cancellation_event.set()
        self._shutdown_event.set()
        
        # Wait for workers to finish
        if hasattr(self, '_worker_tasks'):
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
            
        # Wait for stats task to finish
        if hasattr(self, '_stats_task'):
            self._stats_task.cancel()
            try:
                await self._stats_task
            except asyncio.CancelledError:
                pass
        
        self._is_processing = False
        logger.info("Stopped background OCR processing")

    async def queue_job(self, job: OCRJob):
        """Queue an OCR job for background processing"""
        if not self._is_processing:
            raise OCRError("Background processing is not started")
            
        # Use negative priority for PriorityQueue (higher priority = lower value)
        priority = -job.priority
        await self._job_queue.put((priority, job))

    async def get_results(self) -> AsyncGenerator[OCRJobResult, None]:
        """Get completed OCR results from background processing"""
        if not self._is_processing:
            raise OCRError("Background processing is not started")
            
        while True:
            try:
                result = await asyncio.wait_for(self._result_queue.get(), timeout=1.0)
                yield result
            except asyncio.TimeoutError:
                if self._shutdown_event.is_set():
                    break
                continue

    async def _worker_loop(self, worker_name: str):
        """Worker loop for processing OCR jobs"""
        logger.info(f"Started OCR worker: {worker_name}")
        
        while not self._shutdown_event.is_set():
            try:
                # Get job from queue with timeout
                try:
                    priority, job = await asyncio.wait_for(
                        self._job_queue.get(), 
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process job
                result = await self._process_single_job(job)
                
                # Put result in result queue
                await self._result_queue.put(result)
                
                # Mark job as done
                self._job_queue.task_done()
                
            except asyncio.CancelledError:
                logger.info(f"OCR worker cancelled: {worker_name}")
                break
            except Exception as e:
                logger.error(f"OCR worker error ({worker_name}): {e}")
                
        logger.info(f"Stopped OCR worker: {worker_name}")

    async def _process_single_job(self, job: OCRJob) -> OCRJobResult:
        """Process a single OCR job"""
        start_time = time.time()
        
        try:
            # Check for cancellation
            if self._cancellation_event.is_set():
                return OCRJobResult(
                    job_id=job.job_id,
                    result=OCRResult.empty(),
                    processing_time=0.0,
                    success=False,
                    error_message="Processing cancelled"
                )
            
            # Process OCR in thread pool
            result = await asyncio.get_event_loop().run_in_executor(
                self._thread_pool,
                self._process_ocr_sync,
                job.image,
                job.roi_coords
            )
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self._processing_stats['jobs_completed'] += 1
            self._processing_stats['total_processing_time'] += processing_time
            self._processing_times.append(processing_time)
            
            # Keep only recent processing times
            if len(self._processing_times) > self._max_processing_times:
                self._processing_times.pop(0)
            
            return OCRJobResult(
                job_id=job.job_id,
                result=result,
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._processing_stats['jobs_failed'] += 1
            
            logger.error(f"OCR job failed ({job.job_id}): {e}")
            
            return OCRJobResult(
                job_id=job.job_id,
                result=OCRResult.empty(),
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )

    def _process_ocr_sync(
        self, 
        image: np.ndarray, 
        roi_coords: Optional[Tuple[int, int, int, int]]
    ) -> OCRResult:
        """Synchronous OCR processing for thread pool execution"""
        try:
            # Apply ROI if specified
            if roi_coords:
                x, y, w, h = roi_coords
                image = image[y:y+h, x:x+w]
            
            # Perform OCR
            result = self.ocr_engine.extract_text(image)
            
            # Post-process text
            if result.text.strip():
                processed_text = self.text_processor.post_process_text(result.text)
                result.text = processed_text
            
            return result
            
        except Exception as e:
            raise OCRError(f"OCR processing failed: {e}")

    async def _update_stats_periodically(self):
        """Update performance statistics periodically"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(5.0)  # Update every 5 seconds
                
                if self._processing_times:
                    avg_time = sum(self._processing_times) / len(self._processing_times)
                    self._processing_stats['average_processing_time'] = avg_time
                    
                    # Calculate jobs per second (over last minute)
                    recent_times = [t for t in self._processing_times if time.time() - t < 60]
                    if recent_times:
                        self._processing_stats['jobs_per_second'] = len(recent_times) / 60.0
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Stats update error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        stats = self._processing_stats.copy()
        stats['queue_size'] = self._job_queue.qsize()
        stats['result_queue_size'] = self._result_queue.qsize()
        stats['is_processing'] = self._is_processing
        return stats

    def update_config(self, config: OcrConfig):
        """Update OCR configuration"""
        self.config = config
        self.ocr_engine.update_config(config)
        self.text_processor.update_config(config)

    async def __aenter__(self):
        """Async context manager entry"""
        await self.start_background_processing()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop_background_processing()
        self._thread_pool.shutdown(wait=True)