"""
Service mesh and inter-service communication for VisionSub microservices
"""
import asyncio
import json
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
from aiohttp import web, ClientSession
import redis.asyncio as redis
from datetime import datetime, timedelta
import uuid
from concurrent.futures import ThreadPoolExecutor
import weakref


class ServiceEventType(Enum):
    """Service event types"""
    SERVICE_REGISTERED = "service_registered"
    SERVICE_DEREGISTERED = "service_deregistered"
    SERVICE_HEALTHY = "service_healthy"
    SERVICE_UNHEALTHY = "service_unhealthy"
    CONFIGURATION_UPDATED = "configuration_updated"
    SCALE_EVENT = "scale_event"


@dataclass
class ServiceEvent:
    """Service event"""
    event_id: str
    event_type: ServiceEventType
    service_name: str
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)
    source_service: Optional[str] = None


@dataclass
class ServiceInstance:
    """Service instance information"""
    instance_id: str
    service_name: str
    host: str
    port: int
    version: str
    health_check_url: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_at: datetime = field(default_factory=datetime.utcnow)
    last_health_check: Optional[datetime] = None
    status: str = "unknown"  # healthy, unhealthy, unknown


@dataclass
class ServiceMessage:
    """Inter-service message"""
    message_id: str
    source_service: str
    target_service: Optional[str]  # None for broadcast
    message_type: str
    payload: Dict[str, Any]
    timestamp: datetime
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None


class ServiceRegistry:
    """Service registry for service discovery"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.services: Dict[str, List[ServiceInstance]] = {}
        self.event_handlers: Dict[ServiceEventType, List[Callable]] = {}
        self.health_check_interval = 30
    
    async def register_service(self, instance: ServiceInstance):
        """Register a service instance"""
        # Add to local registry
        if instance.service_name not in self.services:
            self.services[instance.service_name] = []
        
        # Remove existing instance with same ID
        self.services[instance.service_name] = [
            s for s in self.services[instance.service_name] 
            if s.instance_id != instance.instance_id
        ]
        
        self.services[instance.service_name].append(instance)
        
        # Store in Redis
        service_key = f"service:{instance.service_name}:{instance.instance_id}"
        service_data = {
            "instance_id": instance.instance_id,
            "service_name": instance.service_name,
            "host": instance.host,
            "port": instance.port,
            "version": instance.version,
            "health_check_url": instance.health_check_url,
            "metadata": instance.metadata,
            "registered_at": instance.registered_at.isoformat(),
            "last_health_check": instance.last_health_check.isoformat() if instance.last_health_check else None,
            "status": instance.status
        }
        
        await self.redis.hset("service_instances", service_key, json.dumps(service_data))
        await self.redis.sadd(f"service_names:{instance.service_name}", instance.instance_id)
        
        # Emit event
        await self.emit_event(ServiceEvent(
            event_id=str(uuid.uuid4()),
            event_type=ServiceEventType.SERVICE_REGISTERED,
            service_name=instance.service_name,
            timestamp=datetime.utcnow(),
            data=instance.__dict__,
            source_service=instance.service_name
        ))
        
        logging.info(f"Service {instance.service_name} instance {instance.instance_id} registered")
    
    async def deregister_service(self, service_name: str, instance_id: str):
        """Deregister a service instance"""
        if service_name in self.services:
            self.services[service_name] = [
                s for s in self.services[service_name] 
                if s.instance_id != instance_id
            ]
            
            if not self.services[service_name]:
                del self.services[service_name]
        
        # Remove from Redis
        service_key = f"service:{service_name}:{instance_id}"
        await self.redis.hdel("service_instances", service_key)
        await self.redis.srem(f"service_names:{service_name}", instance_id)
        
        # Emit event
        await self.emit_event(ServiceEvent(
            event_id=str(uuid.uuid4()),
            event_type=ServiceEventType.SERVICE_DEREGISTERED,
            service_name=service_name,
            timestamp=datetime.utcnow(),
            data={"instance_id": instance_id},
            source_service=service_name
        ))
        
        logging.info(f"Service {service_name} instance {instance_id} deregistered")
    
    async def get_service_instances(self, service_name: str) -> List[ServiceInstance]:
        """Get all instances of a service"""
        return self.services.get(service_name, [])
    
    async def get_healthy_instances(self, service_name: str) -> List[ServiceInstance]:
        """Get healthy instances of a service"""
        instances = await self.get_service_instances(service_name)
        return [instance for instance in instances if instance.status == "healthy"]
    
    async def discover_services(self) -> Dict[str, List[ServiceInstance]]:
        """Discover all services"""
        return self.services.copy()
    
    async def health_check_services(self):
        """Perform health check on all services"""
        for service_name, instances in self.services.items():
            for instance in instances:
                await self._check_instance_health(instance)
    
    async def _check_instance_health(self, instance: ServiceInstance):
        """Check health of a single instance"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(instance.health_check_url, timeout=5) as response:
                    if response.status == 200:
                        old_status = instance.status
                        instance.status = "healthy"
                        instance.last_health_check = datetime.utcnow()
                        
                        if old_status != "healthy":
                            await self.emit_event(ServiceEvent(
                                event_id=str(uuid.uuid4()),
                                event_type=ServiceEventType.SERVICE_HEALTHY,
                                service_name=instance.service_name,
                                timestamp=datetime.utcnow(),
                                data=instance.__dict__,
                                source_service=instance.service_name
                            ))
                    else:
                        old_status = instance.status
                        instance.status = "unhealthy"
                        instance.last_health_check = datetime.utcnow()
                        
                        if old_status != "unhealthy":
                            await self.emit_event(ServiceEvent(
                                event_id=str(uuid.uuid4()),
                                event_type=ServiceEventType.SERVICE_UNHEALTHY,
                                service_name=instance.service_name,
                                timestamp=datetime.utcnow(),
                                data=instance.__dict__,
                                source_service=instance.service_name
                            ))
        except Exception as e:
            old_status = instance.status
            instance.status = "unhealthy"
            instance.last_health_check = datetime.utcnow()
            
            if old_status != "unhealthy":
                await self.emit_event(ServiceEvent(
                    event_id=str(uuid.uuid4()),
                    event_type=ServiceEventType.SERVICE_UNHEALTHY,
                    service_name=instance.service_name,
                    timestamp=datetime.utcnow(),
                    data={"error": str(e), "instance_id": instance.instance_id},
                    source_service=instance.service_name
                ))
        
        # Update Redis
        service_key = f"service:{instance.service_name}:{instance.instance_id}"
        service_data = {
            "instance_id": instance.instance_id,
            "service_name": instance.service_name,
            "host": instance.host,
            "port": instance.port,
            "version": instance.version,
            "health_check_url": instance.health_check_url,
            "metadata": instance.metadata,
            "registered_at": instance.registered_at.isoformat(),
            "last_health_check": instance.last_health_check.isoformat() if instance.last_health_check else None,
            "status": instance.status
        }
        await self.redis.hset("service_instances", service_key, json.dumps(service_data))
    
    def add_event_handler(self, event_type: ServiceEventType, handler: Callable):
        """Add event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    async def emit_event(self, event: ServiceEvent):
        """Emit service event"""
        # Store in Redis
        event_data = {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "service_name": event.service_name,
            "timestamp": event.timestamp.isoformat(),
            "data": event.data,
            "source_service": event.source_service
        }
        
        await self.redis.lpush("service_events", json.dumps(event_data))
        await self.redis.ltrim("service_events", 0, 1000)  # Keep last 1000 events
        
        # Call local handlers
        if event.event_type in self.event_handlers:
            for handler in self.event_handlers[event.event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    logging.error(f"Error in event handler: {e}")
    
    async def load_from_redis(self):
        """Load service registry from Redis"""
        service_instances = await self.redis.hgetall("service_instances")
        
        for service_key, service_data_json in service_instances.items():
            try:
                service_data = json.loads(service_data_json)
                instance = ServiceInstance(
                    instance_id=service_data["instance_id"],
                    service_name=service_data["service_name"],
                    host=service_data["host"],
                    port=service_data["port"],
                    version=service_data["version"],
                    health_check_url=service_data["health_check_url"],
                    metadata=service_data.get("metadata", {}),
                    registered_at=datetime.fromisoformat(service_data["registered_at"]),
                    last_health_check=datetime.fromisoformat(service_data["last_health_check"]) if service_data.get("last_health_check") else None,
                    status=service_data.get("status", "unknown")
                )
                
                if instance.service_name not in self.services:
                    self.services[instance.service_name] = []
                self.services[instance.service_name].append(instance)
                
            except Exception as e:
                logging.error(f"Error loading service {service_key}: {e}")
        
        logging.info(f"Loaded {len(service_instances)} service instances from Redis")
    
    async def start_health_monitoring(self):
        """Start continuous health monitoring"""
        while True:
            try:
                await self.health_check_services()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                logging.error(f"Health monitoring error: {e}")
                await asyncio.sleep(5)


class ServiceMesh:
    """Service mesh for inter-service communication"""
    
    def __init__(self, redis_client: redis.Redis, service_name: str):
        self.redis = redis_client
        self.service_name = service_name
        self.registry = ServiceRegistry(redis_client)
        self.session: Optional[ClientSession] = None
        self.message_handlers: Dict[str, Callable] = {}
        self.message_queue = asyncio.Queue()
        self.subscriber = None
        self.publisher = None
        self.load_balancer = RoundRobinLoadBalancer()
        self.circuit_breaker = CircuitBreaker()
        self.retry_policy = RetryPolicy()
        self.message_id_counter = 0
    
    async def initialize(self):
        """Initialize the service mesh"""
        self.session = ClientSession()
        
        # Setup Redis pub/sub
        self.subscriber = self.redis.pubsub()
        await self.subscriber.subscribe(f"service_messages:{self.service_name}")
        
        # Load existing services
        await self.registry.load_from_redis()
        
        # Start message processing
        asyncio.create_task(self._process_messages())
        asyncio.create_task(self._handle_incoming_messages())
        
        logging.info(f"Service mesh initialized for {self.service_name}")
    
    async def register_instance(self, host: str, port: int, version: str, metadata: Dict[str, Any] = None):
        """Register this service instance"""
        instance = ServiceInstance(
            instance_id=f"{self.service_name}_{uuid.uuid4().hex[:8]}",
            service_name=self.service_name,
            host=host,
            port=port,
            version=version,
            health_check_url=f"http://{host}:{port}/health",
            metadata=metadata or {}
        )
        
        await self.registry.register_service(instance)
        return instance.instance_id
    
    async def deregister_instance(self, instance_id: str):
        """Deregister this service instance"""
        await self.registry.deregister_service(self.service_name, instance_id)
    
    async def send_message(self, target_service: str, message_type: str, payload: Dict[str, Any], 
                          correlation_id: str = None, reply_to: str = None) -> str:
        """Send message to another service"""
        self.message_id_counter += 1
        message_id = f"{self.service_name}_{self.message_id_counter}"
        
        message = ServiceMessage(
            message_id=message_id,
            source_service=self.service_name,
            target_service=target_service,
            message_type=message_type,
            payload=payload,
            timestamp=datetime.utcnow(),
            correlation_id=correlation_id,
            reply_to=reply_to
        )
        
        # Publish to Redis
        await self.redis.publish(
            f"service_messages:{target_service}",
            json.dumps({
                "message_id": message.message_id,
                "source_service": message.source_service,
                "target_service": message.target_service,
                "message_type": message.message_type,
                "payload": message.payload,
                "timestamp": message.timestamp.isoformat(),
                "correlation_id": message.correlation_id,
                "reply_to": message.reply_to
            })
        )
        
        return message_id
    
    async def broadcast_message(self, message_type: str, payload: Dict[str, Any]):
        """Broadcast message to all services"""
        # Get all service names
        services = await self.registry.discover_services()
        
        for service_name in services:
            if service_name != self.service_name:
                await self.send_message(service_name, message_type, payload)
    
    async def request_response(self, target_service: str, message_type: str, payload: Dict[str, Any], 
                             timeout: float = 30.0) -> Optional[Dict[str, Any]]:
        """Send request and wait for response"""
        correlation_id = str(uuid.uuid4())
        reply_queue = f"reply:{self.service_name}:{correlation_id}"
        
        # Subscribe to reply queue
        reply_subscriber = self.redis.pubsub()
        await reply_subscriber.subscribe(reply_queue)
        
        # Send request
        await self.send_message(target_service, message_type, payload, correlation_id, reply_queue)
        
        # Wait for response
        try:
            start_time = asyncio.get_event_loop().time()
            while True:
                message = await reply_subscriber.get_message(timeout=timeout)
                if message and message["type"] == "message":
                    response = json.loads(message["data"])
                    return response
                
                # Check timeout
                if asyncio.get_event_loop().time() - start_time > timeout:
                    break
                
                await asyncio.sleep(0.1)
                
        except asyncio.TimeoutError:
            logging.warning(f"Request to {target_service} timed out")
        finally:
            await reply_subscriber.unsubscribe(reply_queue)
            await reply_subscriber.close()
        
        return None
    
    async def call_service(self, service_name: str, endpoint: str, method: str = "GET", 
                          payload: Dict[str, Any] = None, timeout: float = 30.0) -> Optional[Dict[str, Any]]:
        """Make HTTP call to another service with load balancing and circuit breaker"""
        # Get healthy instances
        instances = await self.registry.get_healthy_instances(service_name)
        
        if not instances:
            logging.error(f"No healthy instances found for service {service_name}")
            return None
        
        # Load balancing
        instance = self.load_balancer.select_instance(instances)
        
        if not instance:
            logging.error(f"Load balancer failed for service {service_name}")
            return None
        
        # Circuit breaker check
        service_key = f"{service_name}:{instance.instance_id}"
        if not self.circuit_breaker.is_allowed(service_key):
            logging.warning(f"Circuit breaker open for {service_key}")
            return None
        
        # Make the call
        url = f"http://{instance.host}:{instance.port}/{endpoint.lstrip('/')}"
        
        try:
            async with self.session.request(
                method=method,
                url=url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    self.circuit_breaker.record_success(service_key)
                    return result
                else:
                    self.circuit_breaker.record_failure(service_key)
                    logging.error(f"Service {service_name} returned status {response.status}")
                    return None
                    
        except Exception as e:
            self.circuit_breaker.record_failure(service_key)
            logging.error(f"Error calling service {service_name}: {e}")
            
            # Retry with exponential backoff
            return await self.retry_policy.execute(
                lambda: self._retry_service_call(service_name, endpoint, method, payload, timeout),
                max_attempts=3
            )
    
    async def _retry_service_call(self, service_name: str, endpoint: str, method: str, 
                                 payload: Dict[str, Any], timeout: float) -> Optional[Dict[str, Any]]:
        """Retry service call"""
        instances = await self.registry.get_healthy_instances(service_name)
        if not instances:
            return None
        
        instance = self.load_balancer.select_instance(instances)
        if not instance:
            return None
        
        url = f"http://{instance.host}:{instance.port}/{endpoint.lstrip('/')}"
        
        try:
            async with self.session.request(
                method=method,
                url=url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return None
        except Exception:
            return None
    
    def add_message_handler(self, message_type: str, handler: Callable):
        """Add message handler"""
        self.message_handlers[message_type] = handler
    
    async def _process_messages(self):
        """Process outgoing messages"""
        while True:
            try:
                message = await self.message_queue.get()
                # Process message logic here
                self.message_queue.task_done()
            except Exception as e:
                logging.error(f"Error processing message: {e}")
    
    async def _handle_incoming_messages(self):
        """Handle incoming messages"""
        async for message in self.subscriber.listen():
            if message["type"] == "message":
                try:
                    data = json.loads(message["data"])
                    message_type = data.get("message_type")
                    
                    if message_type in self.message_handlers:
                        handler = self.message_handlers[message_type]
                        
                        if asyncio.iscoroutinefunction(handler):
                            await handler(data)
                        else:
                            handler(data)
                            
                except Exception as e:
                    logging.error(f"Error handling message: {e}")
    
    async def get_service_metrics(self) -> Dict[str, Any]:
        """Get service mesh metrics"""
        services = await self.registry.discover_services()
        
        metrics = {
            "total_services": len(services),
            "service_status": {},
            "circuit_breaker_status": {},
            "message_queue_size": self.message_queue.qsize()
        }
        
        for service_name, instances in services.items():
            healthy_count = len([i for i in instances if i.status == "healthy"])
            metrics["service_status"][service_name] = {
                "total_instances": len(instances),
                "healthy_instances": healthy_count,
                "health_percentage": (healthy_count / len(instances)) * 100 if instances else 0
            }
        
        return metrics
    
    async def close(self):
        """Close service mesh connections"""
        if self.session:
            await self.session.close()
        if self.subscriber:
            await self.subscriber.close()
        logging.info(f"Service mesh closed for {self.service_name}")


class RoundRobinLoadBalancer:
    """Round-robin load balancer"""
    
    def __init__(self):
        self.counters = {}
    
    def select_instance(self, instances: List[ServiceInstance]) -> Optional[ServiceInstance]:
        """Select instance using round-robin"""
        if not instances:
            return None
        
        healthy_instances = [i for i in instances if i.status == "healthy"]
        if not healthy_instances:
            return None
        
        service_name = healthy_instances[0].service_name
        if service_name not in self.counters:
            self.counters[service_name] = 0
        
        index = self.counters[service_name] % len(healthy_instances)
        self.counters[service_name] += 1
        
        return healthy_instances[index]


class CircuitBreaker:
    """Circuit breaker implementation"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = {}
        self.last_failure_time = {}
        self.state = {}  # closed, open, half_open
    
    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed"""
        now = datetime.utcnow()
        
        if key not in self.state:
            self.state[key] = "closed"
            self.failure_count[key] = 0
        
        if self.state[key] == "closed":
            return True
        elif self.state[key] == "open":
            if key in self.last_failure_time:
                if (now - self.last_failure_time[key]).total_seconds() > self.recovery_timeout:
                    self.state[key] = "half_open"
                    return True
            return False
        else:  # half_open
            return True
    
    def record_success(self, key: str):
        """Record successful call"""
        self.failure_count[key] = 0
        self.state[key] = "closed"
    
    def record_failure(self, key: str):
        """Record failed call"""
        self.failure_count[key] += 1
        self.last_failure_time[key] = datetime.utcnow()
        
        if self.failure_count[key] >= self.failure_threshold:
            self.state[key] = "open"


class RetryPolicy:
    """Retry policy with exponential backoff"""
    
    def __init__(self, base_delay: float = 1.0, max_delay: float = 10.0):
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    async def execute(self, operation: Callable, max_attempts: int = 3) -> Any:
        """Execute operation with retry"""
        for attempt in range(max_attempts):
            try:
                result = operation()
                if asyncio.iscoroutine(result):
                    result = await result
                return result
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise e
                
                delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                await asyncio.sleep(delay)
        
        return None