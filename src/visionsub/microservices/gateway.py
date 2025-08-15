"""
Service Gateway and API Gateway for VisionSub microservices
"""
import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import aiohttp
from aiohttp import web, ClientSession
from pydantic import BaseModel, Field
import redis.asyncio as redis
from datetime import datetime, timedelta
import jwt
from functools import wraps


@dataclass
class GatewayConfig:
    """Configuration for API Gateway"""
    host: str = "0.0.0.0"
    port: int = 8000
    redis_url: str = "redis://localhost:6379"
    jwt_secret: str = "your-secret-key-change-in-production"
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    cors_origins: List[str] = None
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["*"]


class GatewayRequest(BaseModel):
    """Gateway request model"""
    path: str
    method: str
    headers: Dict[str, str] = Field(default_factory=dict)
    body: Optional[Dict[str, Any]] = None
    query_params: Dict[str, str] = Field(default_factory=dict)
    client_ip: Optional[str] = None


class GatewayResponse(BaseModel):
    """Gateway response model"""
    status_code: int
    headers: Dict[str, str] = Field(default_factory=dict)
    body: Optional[Dict[str, Any]] = None
    service_name: str
    response_time: float


class RateLimiter:
    """Rate limiter implementation using Redis"""
    
    def __init__(self, redis_client: redis.Redis, window_seconds: int = 60, max_requests: int = 100):
        self.redis = redis_client
        self.window = window_seconds
        self.max_requests = max_requests
    
    async def is_allowed(self, key: str) -> tuple[bool, Dict[str, Any]]:
        """Check if request is allowed"""
        pipe = self.redis.pipeline()
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=self.window)
        
        # Remove old entries
        pipe.zremrangebyscore(f"rate_limit:{key}", 0, window_start.timestamp())
        
        # Add current request
        pipe.zadd(f"rate_limit:{key}", {now.timestamp(): now.timestamp()})
        
        # Count requests in window
        pipe.zcard(f"rate_limit:{key}")
        
        # Set expiration
        pipe.expire(f"rate_limit:{key}", self.window)
        
        results = await pipe.execute()
        count = results[2]
        
        return count <= self.max_requests, {
            "current_count": count,
            "max_requests": self.max_requests,
            "window_seconds": self.window,
            "reset_time": (now + timedelta(seconds=self.window)).isoformat()
        }


class CircuitBreaker:
    """Circuit breaker for service protection"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = {}
        self.last_failure_time = {}
        self.state = {}  # "closed", "open", "half-open"
    
    def is_allowed(self, service: str) -> bool:
        """Check if service call is allowed"""
        now = datetime.utcnow()
        
        if service not in self.state:
            self.state[service] = "closed"
            self.failure_count[service] = 0
        
        if self.state[service] == "closed":
            return True
        elif self.state[service] == "open":
            # Check if recovery timeout has passed
            if service in self.last_failure_time:
                if (now - self.last_failure_time[service]).total_seconds() > self.recovery_timeout:
                    self.state[service] = "half-open"
                    return True
            return False
        else:  # half-open
            return True
    
    def record_success(self, service: str):
        """Record successful service call"""
        self.failure_count[service] = 0
        self.state[service] = "closed"
    
    def record_failure(self, service: str):
        """Record failed service call"""
        self.failure_count[service] += 1
        self.last_failure_time[service] = datetime.utcnow()
        
        if self.failure_count[service] >= self.failure_threshold:
            self.state[service] = "open"


class ServiceDiscovery:
    """Service discovery and health monitoring"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.services = {}
        self.health_check_interval = 30  # seconds
    
    async def register_service(self, name: str, url: str, health_check_url: str):
        """Register a service"""
        service_data = {
            "name": name,
            "url": url,
            "health_check_url": health_check_url,
            "registered_at": datetime.utcnow().isoformat(),
            "last_health_check": None,
            "status": "unknown"
        }
        
        await self.redis.hset("services", name, json.dumps(service_data))
        await self.redis.sadd("active_services", name)
        
        logging.info(f"Service {name} registered at {url}")
    
    async def discover_services(self) -> Dict[str, Any]:
        """Discover all registered services"""
        services = {}
        
        service_names = await self.redis.smembers("active_services")
        for name in service_names:
            service_data = await self.redis.hget("services", name)
            if service_data:
                services[name] = json.loads(service_data)
        
        return services
    
    async def health_check_services(self):
        """Perform health check on all services"""
        services = await self.discover_services()
        
        for name, service_data in services.items():
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(service_data["health_check_url"], timeout=5) as response:
                        if response.status == 200:
                            service_data["status"] = "healthy"
                            service_data["last_health_check"] = datetime.utcnow().isoformat()
                        else:
                            service_data["status"] = "unhealthy"
            except:
                service_data["status"] = "unhealthy"
            
            await self.redis.hset("services", name, json.dumps(service_data))
    
    async def get_service_url(self, service_name: str) -> Optional[str]:
        """Get service URL by name"""
        service_data = await self.redis.hget("services", service_name)
        if service_data:
            return json.loads(service_data)["url"]
        return None
    
    async def start_health_monitoring(self):
        """Start continuous health monitoring"""
        while True:
            try:
                await self.health_check_services()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                logging.error(f"Health monitoring error: {e}")
                await asyncio.sleep(5)


class APIGateway:
    """API Gateway for VisionSub microservices"""
    
    def __init__(self, config: GatewayConfig):
        self.config = config
        self.app = web.Application()
        self.redis_client: Optional[redis.Redis] = None
        self.rate_limiter: Optional[RateLimiter] = None
        self.circuit_breaker = CircuitBreaker()
        self.service_discovery: Optional[ServiceDiscovery] = None
        self.session: Optional[ClientSession] = None
        self.setup_routes()
    
    def setup_routes(self):
        """Setup gateway routes"""
        self.app.add_routes([
            web.get('/health', self.health_check),
            web.get('/metrics', self.get_metrics),
            web.get('/services', self.list_services),
            web.get('/service/{service_name}/health', self.service_health_check),
            web.route('*', '/{service_name}/{path:.*}', self.proxy_request),
            web.get('/', self.dashboard),
        ])
        
        # Add CORS middleware
        self.app.middlewares.append(self.cors_middleware)
    
    async def cors_middleware(self, request: web.Request, handler):
        """CORS middleware"""
        if request.method == "OPTIONS":
            response = web.Response()
        else:
            response = await handler(request)
        
        # Add CORS headers
        response.headers['Access-Control-Allow-Origin'] = ','.join(self.config.cors_origins)
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Access-Control-Max-Age'] = '86400'
        
        return response
    
    async def setup_redis(self):
        """Setup Redis connection"""
        self.redis_client = redis.from_url(self.config.redis_url)
        self.rate_limiter = RateLimiter(
            self.redis_client,
            self.config.rate_limit_window,
            self.config.rate_limit_requests
        )
        self.service_discovery = ServiceDiscovery(self.redis_client)
    
    async def setup_session(self):
        """Setup HTTP session"""
        self.session = ClientSession()
    
    async def health_check(self, request: web.Request) -> web.Response:
        """Gateway health check"""
        health = {
            "service": "api_gateway",
            "status": "healthy",
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {}
        }
        
        if self.service_discovery:
            services = await self.service_discovery.discover_services()
            health["services"] = {name: data["status"] for name, data in services.items()}
        
        return web.json_response(health)
    
    async def get_metrics(self, request: web.Request) -> web.Response:
        """Get gateway metrics"""
        metrics = {
            "gateway": {
                "uptime": "N/A",  # TODO: Implement uptime tracking
                "total_requests": 0,  # TODO: Implement request counting
                "active_connections": 0,  # TODO: Implement connection tracking
            },
            "rate_limiter": {},
            "circuit_breaker": {}
        }
        
        if self.service_discovery:
            services = await self.service_discovery.discover_services()
            metrics["services"] = {
                name: {
                    "status": data["status"],
                    "url": data["url"]
                }
                for name, data in services.items()
            }
        
        return web.json_response(metrics)
    
    async def list_services(self, request: web.Request) -> web.Response:
        """List all registered services"""
        if not self.service_discovery:
            return web.json_response({"error": "Service discovery not available"}, status=503)
        
        services = await self.service_discovery.discover_services()
        return web.json_response({"services": services})
    
    async def service_health_check(self, request: web.Request) -> web.Response:
        """Check specific service health"""
        service_name = request.match_info['service_name']
        
        if not self.service_discovery:
            return web.json_response({"error": "Service discovery not available"}, status=503)
        
        service_url = await self.service_discovery.get_service_url(service_name)
        if not service_url:
            return web.json_response({"error": f"Service {service_name} not found"}, status=404)
        
        try:
            async with self.session.get(f"{service_url}/health", timeout=5) as response:
                if response.status == 200:
                    health = await response.json()
                    return web.json_response(health)
                else:
                    return web.json_response(
                        {"error": f"Service {service_name} returned status {response.status}"},
                        status=response.status
                    )
        except Exception as e:
            return web.json_response(
                {"error": f"Service {service_name} health check failed: {str(e)}"},
                status=503
            )
    
    async def proxy_request(self, request: web.Request) -> web.Response:
        """Proxy request to appropriate service"""
        service_name = request.match_info['service_name']
        path = request.match_info['path']
        
        # Rate limiting
        if self.rate_limiter:
            client_ip = request.remote or "unknown"
            rate_key = f"{client_ip}:{service_name}"
            
            allowed, rate_info = await self.rate_limiter.is_allowed(rate_key)
            if not allowed:
                return web.json_response(
                    {
                        "error": "Rate limit exceeded",
                        "rate_limit_info": rate_info
                    },
                    status=429
                )
        
        # Service discovery
        if not self.service_discovery:
            return web.json_response({"error": "Service discovery not available"}, status=503)
        
        service_url = await self.service_discovery.get_service_url(service_name)
        if not service_url:
            return web.json_response({"error": f"Service {service_name} not found"}, status=404)
        
        # Circuit breaker
        if not self.circuit_breaker.is_allowed(service_name):
            return web.json_response(
                {"error": f"Service {service_name} is currently unavailable"},
                status=503
            )
        
        # Proxy the request
        try:
            target_url = f"{service_url}/{path}"
            
            # Prepare request data
            body = None
            if request.body_exists:
                body = await request.json()
            
            # Make the request
            start_time = asyncio.get_event_loop().time()
            
            async with self.session.request(
                method=request.method,
                url=target_url,
                json=body,
                headers=dict(request.headers),
                params=dict(request.query)
            ) as response:
                response_time = asyncio.get_event_loop().time() - start_time
                
                # Get response body
                response_body = None
                if response.content_type == "application/json":
                    response_body = await response.json()
                else:
                    response_body = await response.text()
                
                # Record success/failure
                if response.status < 500:
                    self.circuit_breaker.record_success(service_name)
                else:
                    self.circuit_breaker.record_failure(service_name)
                
                # Create gateway response
                gateway_response = GatewayResponse(
                    status_code=response.status,
                    headers=dict(response.headers),
                    body=response_body,
                    service_name=service_name,
                    response_time=response_time
                )
                
                # Add gateway headers
                response_headers = dict(response.headers)
                response_headers['X-Gateway-Service'] = service_name
                response_headers['X-Gateway-Response-Time'] = str(response_time)
                
                return web.json_response(
                    response_body,
                    status=response.status,
                    headers=response_headers
                )
                
        except Exception as e:
            self.circuit_breaker.record_failure(service_name)
            return web.json_response(
                {"error": f"Proxy error: {str(e)}"},
                status=503
            )
    
    async def dashboard(self, request: web.Request) -> web.Response:
        """Service dashboard"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>VisionSub API Gateway Dashboard</title>
            <meta charset="utf-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; }
                .header { background: #333; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
                .card { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .status { padding: 5px 10px; border-radius: 4px; font-size: 12px; font-weight: bold; }
                .healthy { background-color: #d4edda; color: #155724; }
                .unhealthy { background-color: #f8d7da; color: #721c24; }
                .unknown { background-color: #fff3cd; color: #856404; }
                table { width: 100%; border-collapse: collapse; margin-top: 10px; }
                th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #f8f9fa; }
                .refresh-btn { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
                .refresh-btn:hover { background: #0056b3; }
                .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }
                .metric-card { background: white; padding: 20px; border-radius: 8px; text-align: center; }
                .metric-value { font-size: 24px; font-weight: bold; color: #007bff; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>VisionSub API Gateway Dashboard</h1>
                    <button class="refresh-btn" onclick="refreshDashboard()">Refresh</button>
                </div>
                
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-value" id="totalServices">-</div>
                        <div>Total Services</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="healthyServices">-</div>
                        <div>Healthy Services</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="gatewayUptime">-</div>
                        <div>Gateway Uptime</div>
                    </div>
                </div>
                
                <div class="card">
                    <h2>Service Status</h2>
                    <table id="serviceTable">
                        <thead>
                            <tr>
                                <th>Service Name</th>
                                <th>Status</th>
                                <th>URL</th>
                                <th>Last Health Check</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody id="serviceTableBody">
                            <tr>
                                <td colspan="5" style="text-align: center;">Loading...</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                
                <div class="card">
                    <h2>Gateway Metrics</h2>
                    <div id="gatewayMetrics">
                        <p>Loading metrics...</p>
                    </div>
                </div>
            </div>
            
            <script>
                function updateServiceTable(services) {
                    const tbody = document.getElementById('serviceTableBody');
                    tbody.innerHTML = '';
                    
                    if (Object.keys(services).length === 0) {
                        tbody.innerHTML = '<tr><td colspan="5" style="text-align: center;">No services found</td></tr>';
                        return;
                    }
                    
                    Object.entries(services).forEach(([name, service]) => {
                        const row = document.createElement('tr');
                        const statusClass = service.status || 'unknown';
                        
                        row.innerHTML = `
                            <td>${name}</td>
                            <td><span class="status ${statusClass}">${statusClass}</span></td>
                            <td>${service.url}</td>
                            <td>${service.last_health_check || 'Never'}</td>
                            <td>
                                <button onclick="checkServiceHealth('${name}')" class="refresh-btn" style="padding: 5px 10px; font-size: 12px;">Check Health</button>
                            </td>
                        `;
                        tbody.appendChild(row);
                    });
                }
                
                function updateMetrics(metrics) {
                    document.getElementById('totalServices').textContent = Object.keys(metrics.services || {}).length;
                    document.getElementById('healthyServices').textContent = 
                        Object.values(metrics.services || {}).filter(s => s.status === 'healthy').length;
                    
                    const metricsHtml = `
                        <p><strong>Gateway Version:</strong> ${metrics.gateway?.version || 'N/A'}</p>
                        <p><strong>Total Requests:</strong> ${metrics.gateway?.total_requests || 'N/A'}</p>
                        <p><strong>Active Connections:</strong> ${metrics.gateway?.active_connections || 'N/A'}</p>
                    `;
                    document.getElementById('gatewayMetrics').innerHTML = metricsHtml;
                }
                
                async function loadDashboard() {
                    try {
                        // Load services
                        const servicesResponse = await fetch('/services');
                        const servicesData = await servicesResponse.json();
                        updateServiceTable(servicesData.services || {});
                        
                        // Load metrics
                        const metricsResponse = await fetch('/metrics');
                        const metricsData = await metricsResponse.json();
                        updateMetrics(metricsData);
                        
                    } catch (error) {
                        console.error('Error loading dashboard:', error);
                        document.getElementById('serviceTableBody').innerHTML = 
                            '<tr><td colspan="5" style="text-align: center;">Error loading data</td></tr>';
                    }
                }
                
                async function checkServiceHealth(serviceName) {
                    try {
                        const response = await fetch(`/service/${serviceName}/health`);
                        const health = await response.json();
                        
                        if (response.ok) {
                            alert(`${serviceName} health check: ${health.status}`);
                        } else {
                            alert(`${serviceName} health check failed: ${health.error}`);
                        }
                    } catch (error) {
                        alert(`Error checking ${serviceName} health: ${error.message}`);
                    }
                }
                
                function refreshDashboard() {
                    loadDashboard();
                }
                
                // Load dashboard on page load
                loadDashboard();
                
                // Auto-refresh every 30 seconds
                setInterval(loadDashboard, 30000);
            </script>
        </body>
        </html>
        """
        return web.Response(text=html, content_type='text/html')
    
    async def start_service_registration(self):
        """Start service registration process"""
        # Register default services
        await self.service_discovery.register_service(
            "ocr_service",
            "http://localhost:8081",
            "http://localhost:8081/health"
        )
        
        await self.service_discovery.register_service(
            "ui_service",
            "http://localhost:8080",
            "http://localhost:8080/health"
        )
    
    async def start(self):
        """Start the API Gateway"""
        await self.setup_redis()
        await self.setup_session()
        
        # Start health monitoring
        if self.service_discovery:
            asyncio.create_task(self.service_discovery.start_health_monitoring())
        
        # Register services
        await self.start_service_registration()
        
        # Start the gateway
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.config.host, self.config.port)
        await site.start()
        
        logging.info(f"API Gateway started on {self.config.host}:{self.config.port}")
        logging.info(f"Dashboard available at: http://{self.config.host}:{self.config.port}")
    
    async def stop(self):
        """Stop the API Gateway"""
        if self.session:
            await self.session.close()
        if self.redis_client:
            await self.redis_client.close()
        logging.info("API Gateway stopped")


async def main():
    """Main function to run the API Gateway"""
    import argparse
    
    parser = argparse.ArgumentParser(description="VisionSub API Gateway")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--redis-url", default="redis://localhost:6379")
    parser.add_argument("--jwt-secret", default="your-secret-key-change-in-production")
    parser.add_argument("--rate-limit", type=int, default=100)
    parser.add_argument("--rate-limit-window", type=int, default=60)
    
    args = parser.parse_args()
    
    config = GatewayConfig(
        host=args.host,
        port=args.port,
        redis_url=args.redis_url,
        jwt_secret=args.jwt_secret,
        rate_limit_requests=args.rate_limit,
        rate_limit_window=args.rate_limit_window
    )
    
    gateway = APIGateway(config)
    await gateway.start()
    
    try:
        # Keep gateway running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logging.info("Shutting down gateway...")
        await gateway.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())