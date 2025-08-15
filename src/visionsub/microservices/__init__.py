"""
Microservices architecture for VisionSub
"""
import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod
import aiohttp
from aiohttp import web
import redis.asyncio as redis
from pydantic import BaseModel, Field


@dataclass
class ServiceConfig:
    """Configuration for individual services"""
    name: str
    host: str
    port: int
    version: str
    description: str
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class OCRRequest(BaseModel):
    """OCR processing request model"""
    image_id: str
    image_data: Optional[str] = None  # Base64 encoded or URL
    config: Dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=0, ge=0, le=10)


class OCRResponse(BaseModel):
    """OCR processing response model"""
    image_id: str
    text: str
    confidence: float
    boxes: List[List[int]]
    processing_time: float
    service_version: str


class ServiceHealth(BaseModel):
    """Service health status"""
    service_name: str
    status: str  # "healthy", "degraded", "unhealthy"
    version: str
    uptime: float
    memory_usage: float
    cpu_usage: float
    last_check: float


class MicroserviceBase(ABC):
    """Base class for all microservices"""
    
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.app = web.Application()
        self.health_check_time = 0
        self.start_time = 0
        self.redis_client: Optional[redis.Redis] = None
        # Note: setup_routes should be called by subclasses
        
    @abstractmethod
    def setup_routes(self):
        """Setup service-specific routes"""
        pass
    
    @abstractmethod
    async def health_check(self) -> ServiceHealth:
        """Service-specific health check"""
        pass
    
    async def setup_redis(self, redis_url: str = "redis://localhost:6379"):
        """Setup Redis connection for inter-service communication"""
        self.redis_client = redis.from_url(redis_url)
    
    async def start_service(self):
        """Start the microservice"""
        self.start_time = asyncio.get_event_loop().time()
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.config.host, self.config.port)
        await site.start()
        logging.info(f"Service {self.config.name} started on {self.config.host}:{self.config.port}")
    
    async def stop_service(self):
        """Stop the microservice"""
        if self.redis_client:
            await self.redis_client.close()
        logging.info(f"Service {self.config.name} stopped")


class OCRService(MicroserviceBase):
    """OCR processing microservice"""
    
    def __init__(self, config: ServiceConfig):
        super().__init__(config)
        self.ocr_engine = None
        self.processing_queue = asyncio.Queue()
        self.workers = []
        self.setup_routes()
        
    def setup_routes(self):
        """Setup OCR service routes"""
        self.app.add_routes([
            web.post('/ocr/process', self.process_ocr),
            web.get('/ocr/status/{job_id}', self.get_job_status),
            web.get('/health', self.health_check_endpoint),
            web.get('/metrics', self.get_metrics),
            web.post('/ocr/batch', self.process_batch),
        ])
    
    async def setup_ocr_engine(self, engine_config: Dict[str, Any]):
        """Setup OCR engine with configuration"""
        # Import here to avoid circular dependencies
        from visionsub.core.async_ocr_engine import AsyncOCREngine
        self.ocr_engine = AsyncOCREngine(engine_config)
    
    async def process_ocr(self, request: web.Request) -> web.Response:
        """Process single OCR request"""
        try:
            data = await request.json()
            ocr_request = OCRRequest(**data)
            
            # Queue the request for processing
            job_id = f"ocr_{ocr_request.image_id}_{asyncio.get_event_loop().time()}"
            await self.processing_queue.put((job_id, ocr_request))
            
            return web.json_response({
                "job_id": job_id,
                "status": "queued",
                "message": "OCR request queued for processing"
            })
            
        except Exception as e:
            return web.json_response(
                {"error": str(e), "status": "error"},
                status=400
            )
    
    async def process_batch(self, request: web.Request) -> web.Response:
        """Process batch OCR requests"""
        try:
            data = await request.json()
            requests = [OCRRequest(**req) for req in data["requests"]]
            
            job_ids = []
            for req in requests:
                job_id = f"ocr_{req.image_id}_{asyncio.get_event_loop().time()}"
                await self.processing_queue.put((job_id, req))
                job_ids.append(job_id)
            
            return web.json_response({
                "batch_id": f"batch_{asyncio.get_event_loop().time()}",
                "job_ids": job_ids,
                "status": "queued",
                "total_requests": len(requests)
            })
            
        except Exception as e:
            return web.json_response(
                {"error": str(e), "status": "error"},
                status=400
            )
    
    async def get_job_status(self, request: web.Request) -> web.Response:
        """Get processing job status"""
        job_id = request.match_info['job_id']
        
        # Check Redis for job status
        if self.redis_client:
            status = await self.redis_client.get(f"job_status:{job_id}")
            if status:
                return web.json_response(json.loads(status))
        
        return web.json_response({
            "job_id": job_id,
            "status": "not_found"
        }, status=404)
    
    async def health_check(self) -> ServiceHealth:
        """OCR service health check"""
        import psutil
        process = psutil.Process()
        
        return ServiceHealth(
            service_name=self.config.name,
            status="healthy" if self.ocr_engine else "degraded",
            version=self.config.version,
            uptime=asyncio.get_event_loop().time() - self.start_time,
            memory_usage=process.memory_info().rss / 1024 / 1024,
            cpu_usage=process.cpu_percent(),
            last_check=asyncio.get_event_loop().time()
        )
    
    async def health_check_endpoint(self, request: web.Request) -> web.Response:
        """Health check endpoint"""
        health = await self.health_check()
        return web.json_response(health.dict())
    
    async def get_metrics(self, request: web.Request) -> web.Response:
        """Get service metrics"""
        metrics = {
            "queue_size": self.processing_queue.qsize(),
            "active_workers": len(self.workers),
            "uptime": asyncio.get_event_loop().time() - self.start_time,
            "service_version": self.config.version
        }
        
        if self.ocr_engine:
            metrics.update(self.ocr_engine.get_metrics())
        
        return web.json_response(metrics)
    
    async def start_workers(self, num_workers: int = 4):
        """Start OCR processing workers"""
        for i in range(num_workers):
            worker = asyncio.create_task(self._worker_loop(f"worker_{i}"))
            self.workers.append(worker)
    
    async def _worker_loop(self, worker_name: str):
        """Worker loop for processing OCR requests"""
        logging.info(f"Worker {worker_name} started")
        
        while True:
            try:
                job_id, ocr_request = await self.processing_queue.get()
                
                # Update job status
                if self.redis_client:
                    await self.redis_client.setex(
                        f"job_status:{job_id}",
                        3600,  # 1 hour TTL
                        json.dumps({
                            "job_id": job_id,
                            "status": "processing",
                            "worker": worker_name,
                            "start_time": asyncio.get_event_loop().time()
                        })
                    )
                
                # Process OCR
                start_time = asyncio.get_event_loop().time()
                
                # Mock OCR processing - replace with actual implementation
                await asyncio.sleep(0.1)  # Simulate processing time
                
                result = OCRResponse(
                    image_id=ocr_request.image_id,
                    text=f"OCR result for {ocr_request.image_id}",
                    confidence=0.95,
                    boxes=[[0, 0, 100, 100]],
                    processing_time=asyncio.get_event_loop().time() - start_time,
                    service_version=self.config.version
                )
                
                # Store result
                if self.redis_client:
                    await self.redis_client.setex(
                        f"job_result:{job_id}",
                        3600,  # 1 hour TTL
                        result.json()
                    )
                    
                    await self.redis_client.setex(
                        f"job_status:{job_id}",
                        3600,
                        json.dumps({
                            "job_id": job_id,
                            "status": "completed",
                            "worker": worker_name,
                            "start_time": start_time,
                            "end_time": asyncio.get_event_loop().time(),
                            "processing_time": result.processing_time
                        })
                    )
                
                self.processing_queue.task_done()
                
            except Exception as e:
                logging.error(f"Worker {worker_name} error: {e}")
                
                # Update job status to failed
                if self.redis_client:
                    await self.redis_client.setex(
                        f"job_status:{job_id}",
                        3600,
                        json.dumps({
                            "job_id": job_id,
                            "status": "failed",
                            "worker": worker_name,
                            "error": str(e)
                        })
                    )


class UIService(MicroserviceBase):
    """UI service microservice"""
    
    def __init__(self, config: ServiceConfig):
        super().__init__(config)
        self.ocr_service_url = None
        self.session = None
        self.setup_routes()
        
    def setup_routes(self):
        """Setup UI service routes"""
        self.app.add_routes([
            web.get('/', self.index),
            web.get('/api/status', self.api_status),
            web.post('/api/ocr/submit', self.submit_ocr),
            web.get('/api/ocr/result/{job_id}', self.get_ocr_result),
            web.get('/health', self.health_check_endpoint),
            web.static('/static', './static'),
        ])
    
    async def setup_session(self):
        """Setup HTTP session for inter-service communication"""
        self.session = aiohttp.ClientSession()
    
    async def set_ocr_service(self, ocr_service_url: str):
        """Set OCR service URL"""
        self.ocr_service_url = ocr_service_url
    
    async def index(self, request: web.Request) -> web.Response:
        """Serve main UI page"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>VisionSub - Video OCR</title>
            <meta charset="utf-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .container { max-width: 800px; margin: 0 auto; }
                .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
                .success { background-color: #d4edda; color: #155724; }
                .error { background-color: #f8d7da; color: #721c24; }
                .info { background-color: #d1ecf1; color: #0c5460; }
                button { padding: 10px 20px; margin: 5px; border: none; border-radius: 5px; cursor: pointer; }
                .primary { background-color: #007bff; color: white; }
                .secondary { background-color: #6c757d; color: white; }
                input[type="file"] { margin: 10px 0; }
                #results { margin-top: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>VisionSub - Video OCR Microservice</h1>
                <div id="status" class="status info">
                    Ready to process OCR requests
                </div>
                
                <div>
                    <h2>Upload Image for OCR</h2>
                    <input type="file" id="imageInput" accept="image/*">
                    <br>
                    <button class="primary" onclick="processImage()">Process OCR</button>
                    <button class="secondary" onclick="checkStatus()">Check Status</button>
                </div>
                
                <div id="results"></div>
            </div>
            
            <script>
                let currentJobId = null;
                
                function updateStatus(message, type = 'info') {
                    const statusDiv = document.getElementById('status');
                    statusDiv.textContent = message;
                    statusDiv.className = `status ${type}`;
                }
                
                async function processImage() {
                    const fileInput = document.getElementById('imageInput');
                    if (!fileInput.files[0]) {
                        updateStatus('Please select an image file', 'error');
                        return;
                    }
                    
                    const file = fileInput.files[0];
                    const reader = new FileReader();
                    
                    reader.onload = async function(e) {
                        try {
                            updateStatus('Processing OCR...', 'info');
                            
                            const response = await fetch('/api/ocr/submit', {
                                method: 'POST',
                                headers: {
                                    'Content-Type': 'application/json',
                                },
                                body: JSON.stringify({
                                    image_data: e.target.result.split(',')[1], // Remove base64 prefix
                                    config: { language: 'zh' }
                                })
                            });
                            
                            const result = await response.json();
                            
                            if (response.ok) {
                                currentJobId = result.job_id;
                                updateStatus(`OCR job queued: ${result.job_id}`, 'success');
                                
                                // Start polling for results
                                pollForResult(result.job_id);
                            } else {
                                updateStatus(`Error: ${result.error}`, 'error');
                            }
                        } catch (error) {
                            updateStatus(`Error: ${error.message}`, 'error');
                        }
                    };
                    
                    reader.readAsDataURL(file);
                }
                
                async function pollForResult(jobId) {
                    const pollInterval = setInterval(async () => {
                        try {
                            const response = await fetch(`/api/ocr/result/${jobId}`);
                            const result = await response.json();
                            
                            if (response.ok) {
                                if (result.status === 'completed') {
                                    clearInterval(pollInterval);
                                    displayResult(result);
                                } else if (result.status === 'failed') {
                                    clearInterval(pollInterval);
                                    updateStatus(`OCR failed: ${result.error}`, 'error');
                                }
                                // Still processing, continue polling
                            } else {
                                clearInterval(pollInterval);
                                updateStatus('Error checking job status', 'error');
                            }
                        } catch (error) {
                            console.error('Error polling for result:', error);
                        }
                    }, 1000); // Poll every second
                }
                
                function displayResult(result) {
                    const resultsDiv = document.getElementById('results');
                    resultsDiv.innerHTML = `
                        <h3>OCR Result</h3>
                        <p><strong>Job ID:</strong> ${result.job_id}</p>
                        <p><strong>Text:</strong> ${result.text}</p>
                        <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(2)}%</p>
                        <p><strong>Processing Time:</strong> ${result.processing_time.toFixed(3)}s</p>
                        <p><strong>Service Version:</strong> ${result.service_version}</p>
                    `;
                    updateStatus('OCR processing completed', 'success');
                }
                
                async function checkStatus() {
                    try {
                        const response = await fetch('/api/status');
                        const status = await response.json();
                        
                        updateStatus(
                            `System Status: ${status.overall_status}<br>` +
                            `OCR Service: ${status.services.ocr}<br>` +
                            `UI Service: ${status.services.ui}`,
                            'info'
                        );
                    } catch (error) {
                        updateStatus(`Error checking status: ${error.message}`, 'error');
                    }
                }
            </script>
        </body>
        </html>
        """
        return web.Response(text=html, content_type='text/html')
    
    async def api_status(self, request: web.Request) -> web.Response:
        """Get API status"""
        status = {
            "overall_status": "healthy",
            "services": {
                "ui": "healthy",
                "ocr": "unknown"
            },
            "timestamp": asyncio.get_event_loop().time()
        }
        
        # Check OCR service status
        if self.ocr_service_url and self.session:
            try:
                async with self.session.get(f"{self.ocr_service_url}/health") as response:
                    if response.status == 200:
                        ocr_health = await response.json()
                        status["services"]["ocr"] = ocr_health.get("status", "unknown")
                    else:
                        status["services"]["ocr"] = "unhealthy"
            except:
                status["services"]["ocr"] = "unavailable"
        
        return web.json_response(status)
    
    async def submit_ocr(self, request: web.Request) -> web.Response:
        """Submit OCR request"""
        try:
            if not self.ocr_service_url:
                return web.json_response(
                    {"error": "OCR service not configured"},
                    status=503
                )
            
            data = await request.json()
            
            # Forward request to OCR service
            async with self.session.post(
                f"{self.ocr_service_url}/ocr/process",
                json=data
            ) as response:
                result = await response.json()
                return web.json_response(result, status=response.status)
                
        except Exception as e:
            return web.json_response(
                {"error": str(e)},
                status=500
            )
    
    async def get_ocr_result(self, request: web.Request) -> web.Response:
        """Get OCR result"""
        try:
            job_id = request.match_info['job_id']
            
            if not self.ocr_service_url:
                return web.json_response(
                    {"error": "OCR service not configured"},
                    status=503
                )
            
            # Forward request to OCR service
            async with self.session.get(
                f"{self.ocr_service_url}/ocr/status/{job_id}"
            ) as response:
                result = await response.json()
                
                # If job is completed, fetch the actual result
                if result.get("status") == "completed" and self.redis_client:
                    result_data = await self.redis_client.get(f"job_result:{job_id}")
                    if result_data:
                        result.update(json.loads(result_data))
                
                return web.json_response(result, status=response.status)
                
        except Exception as e:
            return web.json_response(
                {"error": str(e)},
                status=500
            )
    
    async def health_check(self) -> ServiceHealth:
        """UI service health check"""
        import psutil
        process = psutil.Process()
        
        return ServiceHealth(
            service_name=self.config.name,
            status="healthy",
            version=self.config.version,
            uptime=asyncio.get_event_loop().time() - self.start_time,
            memory_usage=process.memory_info().rss / 1024 / 1024,
            cpu_usage=process.cpu_percent(),
            last_check=asyncio.get_event_loop().time()
        )
    
    async def health_check_endpoint(self, request: web.Request) -> web.Response:
        """Health check endpoint"""
        health = await self.health_check()
        return web.json_response(health.dict())


class ServiceOrchestrator:
    """Orchestrator for managing microservices"""
    
    def __init__(self):
        self.services: Dict[str, MicroserviceBase] = {}
        self.tasks: List[asyncio.Task] = []
        
    def add_service(self, service: MicroserviceBase):
        """Add a service to the orchestrator"""
        self.services[service.config.name] = service
    
    async def start_all_services(self):
        """Start all services"""
        for service in self.services.values():
            task = asyncio.create_task(service.start_service())
            self.tasks.append(task)
    
    async def stop_all_services(self):
        """Stop all services"""
        for task in self.tasks:
            task.cancel()
        
        for service in self.services.values():
            await service.stop_service()
    
    async def setup_redis(self, redis_url: str = "redis://localhost:6379"):
        """Setup Redis for all services"""
        for service in self.services.values():
            await service.setup_redis(redis_url)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        status = {
            "total_services": len(self.services),
            "services": {},
            "timestamp": asyncio.get_event_loop().time()
        }
        
        for name, service in self.services.items():
            try:
                health = await service.health_check()
                status["services"][name] = health.dict()
            except Exception as e:
                status["services"][name] = {
                    "error": str(e),
                    "status": "unavailable"
                }
        
        return status


async def main():
    """Main function to run the microservices"""
    import argparse
    
    parser = argparse.ArgumentParser(description="VisionSub Microservices")
    parser.add_argument("--service", choices=["ocr", "ui", "all"], default="all")
    parser.add_argument("--redis-url", default="redis://localhost:6379")
    parser.add_argument("--ocr-host", default="localhost")
    parser.add_argument("--ocr-port", type=int, default=8081)
    parser.add_argument("--ui-host", default="localhost")
    parser.add_argument("--ui-port", type=int, default=8080)
    
    args = parser.parse_args()
    
    orchestrator = ServiceOrchestrator()
    
    if args.service in ["ocr", "all"]:
        ocr_config = ServiceConfig(
            name="ocr_service",
            host=args.ocr_host,
            port=args.ocr_port,
            version="1.0.0",
            description="OCR Processing Service"
        )
        ocr_service = OCRService(ocr_config)
        orchestrator.add_service(ocr_service)
    
    if args.service in ["ui", "all"]:
        ui_config = ServiceConfig(
            name="ui_service",
            host=args.ui_host,
            port=args.ui_port,
            version="1.0.0",
            description="UI Service"
        )
        ui_service = UIService(ui_config)
        orchestrator.add_service(ui_service)
    
    # Setup Redis
    await orchestrator.setup_redis(args.redis_url)
    
    # Configure service interconnections
    if "ui_service" in orchestrator.services and "ocr_service" in orchestrator.services:
        ui_service = orchestrator.services["ui_service"]
        ui_service.set_ocr_service(f"http://{args.ocr_host}:{args.ocr_port}")
        await ui_service.setup_session()
    
    # Start OCR workers if OCR service is running
    if "ocr_service" in orchestrator.services:
        ocr_service = orchestrator.services["ocr_service"]
        await ocr_service.setup_ocr_engine({})
        await ocr_service.start_workers(4)
    
    # Start all services
    await orchestrator.start_all_services()
    
    logging.info("VisionSub microservices started")
    logging.info(f"UI Service: http://{args.ui_host}:{args.ui_port}")
    logging.info(f"OCR Service: http://{args.ocr_host}:{args.ocr_port}")
    
    try:
        # Keep services running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logging.info("Shutting down services...")
        await orchestrator.stop_all_services()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())