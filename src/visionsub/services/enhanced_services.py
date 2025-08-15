"""
Enhanced Service Layer for VisionSub
"""
import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, AsyncGenerator, Union
from typing import Callable

import numpy as np
from pydantic import BaseModel, Field

from ..core.enhanced_engine import EnhancedProcessingEngine, ProcessingTask
from ..core.enhanced_video_processor import EnhancedVideoProcessor, ProcessingOptions
from ..core.enhanced_ocr_engine import EnhancedOCREngine, OCRSecurityConfig, OCRPerformanceConfig
from ..core.errors import ProcessingError, SecurityError, ConfigurationError
from ..models.config import ProcessingConfig
from ..models.subtitle import SubtitleItem
from ..security.enhanced_security_manager import EnhancedSecurityManager, SecurityContext
from ..utils.metrics import MetricsCollector
from ..utils.health_monitor import HealthMonitor

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service status enumeration"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


class ServicePriority(Enum):
    """Service priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ServiceMetrics:
    """Service metrics"""
    uptime: float = 0.0
    requests_processed: int = 0
    errors_encountered: int = 0
    average_response_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    active_connections: int = 0


@dataclass
class ProcessingRequest:
    """Processing request model"""
    id: str
    video_path: str
    config: ProcessingConfig
    priority: ServicePriority = ServicePriority.NORMAL
    created_at: float = field(default_factory=time.time)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    callback_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingResponse:
    """Processing response model"""
    request_id: str
    status: str
    result: Optional[List[SubtitleItem]] = None
    error: Optional[str] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseService(ABC):
    """Base service class"""

    def __init__(self, name: str, security_manager: EnhancedSecurityManager):
        self.name = name
        self.security_manager = security_manager
        self.status = ServiceStatus.STOPPED
        self.metrics = ServiceMetrics()
        self.start_time: Optional[float] = None
        self._is_running = False
        self._shutdown_event = asyncio.Event()

    @abstractmethod
    async def start(self) -> bool:
        """Start the service"""
        pass

    @abstractmethod
    async def stop(self) -> bool:
        """Stop the service"""
        pass

    @abstractmethod
    async def process_request(self, request: ProcessingRequest) -> ProcessingResponse:
        """Process a request"""
        pass

    async def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        uptime = 0.0
        if self.start_time:
            uptime = time.time() - self.start_time

        return {
            "name": self.name,
            "status": self.status.value,
            "uptime": uptime,
            "metrics": self.metrics.__dict__,
            "is_running": self._is_running
        }

    async def update_metrics(self, **kwargs):
        """Update service metrics"""
        for key, value in kwargs.items():
            if hasattr(self.metrics, key):
                setattr(self.metrics, key, value)


class OCRService(BaseService):
    """Enhanced OCR service"""

    def __init__(self, security_manager: EnhancedSecurityManager):
        super().__init__("ocr_service", security_manager)
        self.ocr_engine: Optional[EnhancedOCREngine] = None
        self.request_queue: asyncio.Queue[ProcessingRequest] = asyncio.Queue()
        self._worker_tasks: List[asyncio.Task] = []
        self.metrics_collector = MetricsCollector()

    async def start(self) -> bool:
        """Start OCR service"""
        try:
            if self.status != ServiceStatus.STOPPED:
                logger.warning(f"Service {self.name} is already running")
                return False

            self.status = ServiceStatus.STARTING
            logger.info(f"Starting {self.name}")

            # Initialize OCR engine
            ocr_config = {}
            security_config = OCRSecurityConfig()
            performance_config = OCRPerformanceConfig()

            self.ocr_engine = EnhancedOCREngine(
                ocr_config, security_config, performance_config
            )

            if not await self.ocr_engine.initialize():
                raise ConfigurationError("Failed to initialize OCR engine")

            # Start worker tasks
            num_workers = 4
            for i in range(num_workers):
                task = asyncio.create_task(self._worker_loop(f"worker-{i}"))
                self._worker_tasks.append(task)

            self.start_time = time.time()
            self.status = ServiceStatus.RUNNING
            self._is_running = True

            logger.info(f"{self.name} started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start {self.name}: {e}")
            self.status = ServiceStatus.ERROR
            return False

    async def stop(self) -> bool:
        """Stop OCR service"""
        try:
            if self.status != ServiceStatus.RUNNING:
                logger.warning(f"Service {self.name} is not running")
                return False

            self.status = ServiceStatus.STOPPING
            logger.info(f"Stopping {self.name}")

            # Signal shutdown
            self._shutdown_event.set()

            # Cancel worker tasks
            for task in self._worker_tasks:
                task.cancel()

            # Wait for tasks to complete
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)

            # Cleanup OCR engine
            if self.ocr_engine:
                self.ocr_engine.cleanup()

            self._worker_tasks.clear()
            self._is_running = False
            self.status = ServiceStatus.STOPPED

            logger.info(f"{self.name} stopped successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to stop {self.name}: {e}")
            self.status = ServiceStatus.ERROR
            return False

    async def process_request(self, request: ProcessingRequest) -> ProcessingResponse:
        """Process OCR request"""
        try:
            start_time = time.time()

            # Validate security context
            if request.session_id:
                context = await self.security_manager.get_security_context(request.session_id)
                if not context:
                    raise SecurityError("Invalid session")
                
                if not await self.security_manager.validate_permission(
                    request.session_id, "ocr:process"
                ):
                    raise SecurityError("Permission denied")

            # Process image
            if request.metadata.get("image_data"):
                image = np.frombuffer(request.metadata["image_data"], dtype=np.uint8)
                image = image.reshape(request.metadata["image_shape"])
                
                result = await self.ocr_engine.process_image_enhanced(image)
                
                # Convert to subtitle items
                subtitles = []
                if result and result.original_result.items:
                    for item in result.original_result.items:
                        subtitle = SubtitleItem(
                            index=len(subtitles),
                            start_time=request.metadata.get("timestamp", 0.0),
                            end_time=request.metadata.get("timestamp", 0.0) + 1.0,
                            text=item.text,
                            confidence=item.confidence
                        )
                        subtitles.append(subtitle)
            else:
                raise ProcessingError("No image data provided")

            processing_time = time.time() - start_time

            # Update metrics
            await self.update_metrics(
                requests_processed=self.metrics.requests_processed + 1,
                average_response_time=(
                    (self.metrics.average_response_time * self.metrics.requests_processed + processing_time) /
                    (self.metrics.requests_processed + 1)
                )
            )

            return ProcessingResponse(
                request_id=request.id,
                status="success",
                result=subtitles,
                processing_time=processing_time,
                metadata={
                    "engine_info": result.engine_info if result else {},
                    "quality_score": result.quality_score if result else 0.0
                }
            )

        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            await self.update_metrics(
                errors_encountered=self.metrics.errors_encountered + 1
            )
            return ProcessingResponse(
                request_id=request.id,
                status="error",
                error=str(e)
            )

    async def _worker_loop(self, worker_name: str):
        """Worker loop for processing requests"""
        try:
            logger.info(f"OCR worker {worker_name} started")

            while not self._shutdown_event.is_set():
                try:
                    # Get request with timeout
                    request = await asyncio.wait_for(
                        self.request_queue.get(), timeout=1.0
                    )

                    # Process request
                    response = await self.process_request(request)

                    # Handle callback if specified
                    if request.callback_url:
                        await self._send_callback(request.callback_url, response)

                    logger.debug(f"Worker {worker_name} processed request {request.id}")

                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Worker {worker_name} error: {e}")

        except asyncio.CancelledError:
            logger.info(f"OCR worker {worker_name} cancelled")
        except Exception as e:
            logger.error(f"Worker {worker_name} failed: {e}")


class VideoProcessingService(BaseService):
    """Enhanced video processing service"""

    def __init__(self, security_manager: EnhancedSecurityManager):
        super().__init__("video_processing_service", security_manager)
        self.processing_engine: Optional[EnhancedProcessingEngine] = None
        self.video_processor: Optional[EnhancedVideoProcessor] = None
        self.request_queue: asyncio.Queue[ProcessingRequest] = asyncio.Queue()
        self._active_requests: Dict[str, asyncio.Task] = {}
        self._worker_tasks: List[asyncio.Task] = []

    async def start(self) -> bool:
        """Start video processing service"""
        try:
            if self.status != ServiceStatus.STOPPED:
                logger.warning(f"Service {self.name} is already running")
                return False

            self.status = ServiceStatus.STARTING
            logger.info(f"Starting {self.name}")

            # Initialize processing engine
            self.processing_engine = EnhancedProcessingEngine()
            await self.processing_engine.start_processing_queue()

            # Initialize video processor
            processing_options = ProcessingOptions()
            self.video_processor = EnhancedVideoProcessor(
                ProcessingConfig(), processing_options
            )

            # Start worker tasks
            num_workers = 2
            for i in range(num_workers):
                task = asyncio.create_task(self._worker_loop(f"worker-{i}"))
                self._worker_tasks.append(task)

            self.start_time = time.time()
            self.status = ServiceStatus.RUNNING
            self._is_running = True

            logger.info(f"{self.name} started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start {self.name}: {e}")
            self.status = ServiceStatus.ERROR
            return False

    async def stop(self) -> bool:
        """Stop video processing service"""
        try:
            if self.status != ServiceStatus.RUNNING:
                logger.warning(f"Service {self.name} is not running")
                return False

            self.status = ServiceStatus.STOPPING
            logger.info(f"Stopping {self.name}")

            # Cancel active requests
            for task in self._active_requests.values():
                task.cancel()

            # Wait for active requests to complete
            await asyncio.gather(*self._active_requests.values(), return_exceptions=True)

            # Cancel worker tasks
            for task in self._worker_tasks:
                task.cancel()

            # Wait for worker tasks to complete
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)

            # Cleanup engines
            if self.processing_engine:
                await self.processing_engine.cleanup()

            if self.video_processor:
                await self.video_processor.cleanup()

            self._active_requests.clear()
            self._worker_tasks.clear()
            self._is_running = False
            self.status = ServiceStatus.STOPPED

            logger.info(f"{self.name} stopped successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to stop {self.name}: {e}")
            self.status = ServiceStatus.ERROR
            return False

    async def process_request(self, request: ProcessingRequest) -> ProcessingResponse:
        """Process video processing request"""
        try:
            start_time = time.time()

            # Validate security context
            if request.session_id:
                context = await self.security_manager.get_security_context(request.session_id)
                if not context:
                    raise SecurityError("Invalid session")
                
                if not await self.security_manager.validate_permission(
                    request.session_id, "video:process"
                ):
                    raise SecurityError("Permission denied")

            # Process video
            result = await self.processing_engine.process_video(
                request.video_path, request.config
            )

            processing_time = time.time() - start_time

            # Update metrics
            await self.update_metrics(
                requests_processed=self.metrics.requests_processed + 1,
                average_response_time=(
                    (self.metrics.average_response_time * self.metrics.requests_processed + processing_time) /
                    (self.metrics.requests_processed + 1)
                )
            )

            return ProcessingResponse(
                request_id=request.id,
                status="success",
                result=result,
                processing_time=processing_time,
                metadata={
                    "video_path": request.video_path,
                    "subtitles_count": len(result) if result else 0
                }
            )

        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            await self.update_metrics(
                errors_encountered=self.metrics.errors_encountered + 1
            )
            return ProcessingResponse(
                request_id=request.id,
                status="error",
                error=str(e)
            )

    async def _worker_loop(self, worker_name: str):
        """Worker loop for processing requests"""
        try:
            logger.info(f"Video processing worker {worker_name} started")

            while not self._shutdown_event.is_set():
                try:
                    # Get request with timeout
                    request = await asyncio.wait_for(
                        self.request_queue.get(), timeout=1.0
                    )

                    # Start processing task
                    task = asyncio.create_task(self._process_request_async(request))
                    self._active_requests[request.id] = task

                    # Handle completion
                    task.add_done_callback(
                        lambda t: self._active_requests.pop(request.id, None)
                    )

                    logger.debug(f"Worker {worker_name} started processing request {request.id}")

                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Worker {worker_name} error: {e}")

        except asyncio.CancelledError:
            logger.info(f"Video processing worker {worker_name} cancelled")
        except Exception as e:
            logger.error(f"Worker {worker_name} failed: {e}")

    async def _process_request_async(self, request: ProcessingRequest):
        """Process request asynchronously"""
        try:
            response = await self.process_request(request)

            # Handle callback if specified
            if request.callback_url:
                await self._send_callback(request.callback_url, response)

        except Exception as e:
            logger.error(f"Async processing failed for request {request.id}: {e}")


class ServiceManager:
    """Service manager for coordinating multiple services"""

    def __init__(self, security_manager: EnhancedSecurityManager):
        self.security_manager = security_manager
        self.services: Dict[str, BaseService] = {}
        self.health_monitor = HealthMonitor()
        self.metrics_collector = MetricsCollector()

    async def register_service(self, service: BaseService):
        """Register a service"""
        self.services[service.name] = service
        logger.info(f"Registered service: {service.name}")

    async def start_all_services(self) -> bool:
        """Start all registered services"""
        try:
            logger.info("Starting all services")

            # Start services in order of priority
            start_order = ["ocr_service", "video_processing_service"]
            
            for service_name in start_order:
                if service_name in self.services:
                    success = await self.services[service_name].start()
                    if not success:
                        logger.error(f"Failed to start service: {service_name}")
                        return False

            logger.info("All services started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start services: {e}")
            return False

    async def stop_all_services(self) -> bool:
        """Stop all registered services"""
        try:
            logger.info("Stopping all services")

            # Stop services in reverse order
            stop_order = ["video_processing_service", "ocr_service"]
            
            for service_name in stop_order:
                if service_name in self.services:
                    success = await self.services[service_name].stop()
                    if not success:
                        logger.error(f"Failed to stop service: {service_name}")

            logger.info("All services stopped")
            return True

        except Exception as e:
            logger.error(f"Failed to stop services: {e}")
            return False

    async def get_service_status(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific service"""
        if service_name in self.services:
            return await self.services[service_name].get_status()
        return None

    async def get_all_services_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all services"""
        status = {}
        for name, service in self.services.items():
            status[name] = await service.get_status()
        return status

    async def process_request(
        self, 
        service_name: str, 
        request: ProcessingRequest
    ) -> ProcessingResponse:
        """Process request through a specific service"""
        if service_name not in self.services:
            raise ProcessingError(f"Service not found: {service_name}")

        return await self.services[service_name].process_request(request)

    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide metrics"""
        try:
            # Collect metrics from all services
            service_metrics = {}
            for name, service in self.services.items():
                service_metrics[name] = service.metrics.__dict__

            return {
                "timestamp": time.time(),
                "services": service_metrics,
                "system_metrics": self.metrics_collector.get_metrics(),
                "health_status": await self.health_monitor.get_health_status()
            }

        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {"error": str(e)}

    async def _send_callback(self, callback_url: str, response: ProcessingResponse):
        """Send callback response"""
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                payload = {
                    "request_id": response.request_id,
                    "status": response.status,
                    "processing_time": response.processing_time,
                    "metadata": response.metadata
                }
                
                if response.result:
                    payload["result"] = [item.dict() for item in response.result]
                
                if response.error:
                    payload["error"] = response.error
                
                async with session.post(callback_url, json=payload) as resp:
                    if resp.status != 200:
                        logger.warning(f"Callback failed: {resp.status}")
                    
        except Exception as e:
            logger.error(f"Callback sending failed: {e}")


# Global service manager instance
_service_manager: Optional[ServiceManager] = None


async def get_service_manager() -> ServiceManager:
    """Get or create global service manager"""
    global _service_manager
    if _service_manager is None:
        security_manager = await get_security_manager()
        _service_manager = ServiceManager(security_manager)
        
        # Register services
        ocr_service = OCRService(security_manager)
        video_service = VideoProcessingService(security_manager)
        
        await _service_manager.register_service(ocr_service)
        await _service_manager.register_service(video_service)
        
        # Start services
        await _service_manager.start_all_services()
    
    return _service_manager


async def process_video_request(
    video_path: str,
    config: ProcessingConfig,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None
) -> ProcessingResponse:
    """Process video request through service manager"""
    service_manager = await get_service_manager()
    
    request = ProcessingRequest(
        id=f"req_{int(time.time() * 1000)}",
        video_path=video_path,
        config=config,
        user_id=user_id,
        session_id=session_id
    )
    
    return await service_manager.process_request("video_processing_service", request)