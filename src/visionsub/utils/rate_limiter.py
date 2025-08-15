"""
Rate Limiter for API Protection
"""
import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Union


@dataclass
class RateLimitInfo:
    """Rate limit information"""
    requests_allowed: bool
    remaining_requests: int
    reset_time: float
    retry_after: Optional[float] = None


class RateLimiter:
    """Token bucket rate limiter with sliding window"""

    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: int = 60,
        enable_burst: bool = True,
        burst_factor: float = 2.0
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.enable_burst = enable_burst
        self.burst_factor = burst_factor
        
        # Calculate burst limit
        self.burst_limit = int(max_requests * burst_factor) if enable_burst else max_requests
        
        # Storage for request timestamps
        self.request_timestamps: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.burst_limit))
        
        # Blacklist for abusive clients
        self.blacklist: Set[str] = set()
        self.blacklist_expiry: Dict[str, float] = {}
        
        # Statistics
        self.total_requests = 0
        self.blocked_requests = 0
        self.limited_requests = 0

    async def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed for client"""
        rate_info = await self.get_rate_limit_info(client_id)
        return rate_info.requests_allowed

    async def get_rate_limit_info(self, client_id: str) -> RateLimitInfo:
        """Get detailed rate limit information"""
        # Check blacklist
        if client_id in self.blacklist:
            expiry_time = self.blacklist_expiry.get(client_id, 0)
            if time.time() < expiry_time:
                return RateLimitInfo(
                    requests_allowed=False,
                    remaining_requests=0,
                    reset_time=expiry_time,
                    retry_after=expiry_time - time.time()
                )
            else:
                # Remove expired blacklist entry
                self.blacklist.remove(client_id)
                del self.blacklist_expiry[client_id]
        
        # Get request timestamps for client
        timestamps = self.request_timestamps[client_id]
        current_time = time.time()
        
        # Remove expired timestamps
        window_start = current_time - self.window_seconds
        while timestamps and timestamps[0] < window_start:
            timestamps.popleft()
        
        # Check if client is rate limited
        requests_in_window = len(timestamps)
        
        if requests_in_window >= self.max_requests:
            # Check if burst is allowed
            if self.enable_burst and requests_in_window < self.burst_limit:
                # Allow burst but track for future limiting
                remaining = max(0, self.burst_limit - requests_in_window)
                reset_time = timestamps[0] + self.window_seconds if timestamps else current_time + self.window_seconds
                
                return RateLimitInfo(
                    requests_allowed=True,
                    remaining_requests=remaining,
                    reset_time=reset_time
                )
            else:
                # Rate limited
                self.limited_requests += 1
                
                # Calculate retry after time
                retry_after = 0
                if timestamps:
                    retry_after = timestamps[0] + self.window_seconds - current_time
                
                return RateLimitInfo(
                    requests_allowed=False,
                    remaining_requests=0,
                    reset_time=timestamps[0] + self.window_seconds if timestamps else current_time + self.window_seconds,
                    retry_after=max(0, retry_after)
                )
        
        # Request allowed
        remaining = max(0, self.max_requests - requests_in_window - 1)
        reset_time = timestamps[0] + self.window_seconds if timestamps else current_time + self.window_seconds
        
        return RateLimitInfo(
            requests_allowed=True,
            remaining_requests=remaining,
            reset_time=reset_time
        )

    async def record_request(self, client_id: str):
        """Record a request for rate limiting"""
        self.total_requests += 1
        
        # Add current timestamp
        current_time = time.time()
        self.request_timestamps[client_id].append(current_time)
        
        # Check for abuse patterns
        await self._check_abuse_patterns(client_id)

    async def check_and_record(self, client_id: str) -> bool:
        """Check rate limit and record request if allowed"""
        rate_info = await self.get_rate_limit_info(client_id)
        
        if rate_info.requests_allowed:
            await self.record_request(client_id)
            return True
        else:
            return False

    async def _check_abuse_patterns(self, client_id: str):
        """Check for abuse patterns and blacklist if necessary"""
        timestamps = self.request_timestamps[client_id]
        
        if len(timestamps) < 10:
            return
        
        # Check for rapid successive requests
        recent_requests = [t for t in timestamps if time.time() - t < 5]  # Last 5 seconds
        if len(recent_requests) > 20:  # More than 20 requests in 5 seconds
            await self._blacklist_client(client_id, duration=300)  # 5 minutes
        
        # Check for sustained high rate
        if len(timestamps) > self.burst_limit * 2:  # Double the burst limit
            await self._blacklist_client(client_id, duration=600)  # 10 minutes

    async def _blacklist_client(self, client_id: str, duration: int):
        """Blacklist a client for specified duration"""
        expiry_time = time.time() + duration
        self.blacklist.add(client_id)
        self.blacklist_expiry[client_id] = expiry_time
        
        # Clear request timestamps
        if client_id in self.request_timestamps:
            del self.request_timestamps[client_id]

    async def remove_from_blacklist(self, client_id: str):
        """Remove client from blacklist"""
        if client_id in self.blacklist:
            self.blacklist.remove(client_id)
        if client_id in self.blacklist_expiry:
            del self.blacklist_expiry[client_id]

    async def get_blacklist(self) -> Dict[str, float]:
        """Get current blacklist with expiry times"""
        return self.blacklist_expiry.copy()

    async def clear_blacklist(self):
        """Clear all blacklist entries"""
        self.blacklist.clear()
        self.blacklist_expiry.clear()

    async def get_client_stats(self, client_id: str) -> Dict[str, Union[int, float]]:
        """Get statistics for a specific client"""
        timestamps = self.request_timestamps.get(client_id, deque())
        
        if not timestamps:
            return {
                "total_requests": 0,
                "requests_in_window": 0,
                "is_blacklisted": client_id in self.blacklist,
                "blacklist_expiry": self.blacklist_expiry.get(client_id, 0)
            }
        
        current_time = time.time()
        window_start = current_time - self.window_seconds
        
        # Count requests in current window
        requests_in_window = sum(1 for t in timestamps if t >= window_start)
        
        # Calculate request rate
        if len(timestamps) > 1:
            time_span = timestamps[-1] - timestamps[0]
            request_rate = len(timestamps) / max(time_span, 1)
        else:
            request_rate = 0
        
        return {
            "total_requests": len(timestamps),
            "requests_in_window": requests_in_window,
            "request_rate_per_second": request_rate,
            "is_blacklisted": client_id in self.blacklist,
            "blacklist_expiry": self.blacklist_expiry.get(client_id, 0)
        }

    async def get_global_stats(self) -> Dict[str, Union[int, float]]:
        """Get global rate limiting statistics"""
        return {
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "limited_requests": self.limited_requests,
            "blacklisted_clients": len(self.blacklist),
            "active_clients": len(self.request_timestamps),
            "max_requests_per_window": self.max_requests,
            "window_seconds": self.window_seconds,
            "burst_limit": self.burst_limit
        }

    async def reset_client(self, client_id: str):
        """Reset rate limiting for a specific client"""
        if client_id in self.request_timestamps:
            del self.request_timestamps[client_id]
        await self.remove_from_blacklist(client_id)

    async def cleanup_expired_entries(self):
        """Cleanup expired blacklist entries and old timestamps"""
        current_time = time.time()
        
        # Cleanup expired blacklist entries
        expired_clients = [
            client_id for client_id, expiry_time in self.blacklist_expiry.items()
            if expiry_time <= current_time
        ]
        
        for client_id in expired_clients:
            await self.remove_from_blacklist(client_id)
        
        # Cleanup old timestamps for inactive clients
        window_start = current_time - (self.window_seconds * 2)  # Keep for 2 windows
        inactive_clients = []
        
        for client_id, timestamps in self.request_timestamps.items():
            if not timestamps or timestamps[-1] < window_start:
                inactive_clients.append(client_id)
        
        for client_id in inactive_clients:
            del self.request_timestamps[client_id]

    def update_settings(
        self,
        max_requests: Optional[int] = None,
        window_seconds: Optional[int] = None,
        enable_burst: Optional[bool] = None,
        burst_factor: Optional[float] = None
    ):
        """Update rate limiter settings"""
        if max_requests is not None:
            self.max_requests = max_requests
        if window_seconds is not None:
            self.window_seconds = window_seconds
        if enable_burst is not None:
            self.enable_burst = enable_burst
        if burst_factor is not None:
            self.burst_factor = burst_factor
        
        # Recalculate burst limit
        self.burst_limit = int(self.max_requests * self.burst_factor) if self.enable_burst else self.max_requests


class DistributedRateLimiter:
    """Distributed rate limiter using Redis (placeholder implementation)"""

    def __init__(
        self,
        redis_client=None,
        max_requests: int = 100,
        window_seconds: int = 60
    ):
        self.redis_client = redis_client
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        
        # Fallback to local rate limiter if Redis not available
        self.local_limiter = RateLimiter(max_requests, window_seconds)

    async def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed"""
        if self.redis_client is None:
            return await self.local_limiter.is_allowed(client_id)
        
        # Redis implementation would go here
        # For now, fall back to local limiter
        return await self.local_limiter.is_allowed(client_id)

    async def record_request(self, client_id: str):
        """Record a request"""
        if self.redis_client is None:
            await self.local_limiter.record_request(client_id)
        else:
            # Redis implementation would go here
            await self.local_limiter.record_request(client_id)


# Decorator for rate limiting
def rate_limit(max_requests: int = 100, window_seconds: int = 60):
    """Decorator for rate limiting functions"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Get client ID from kwargs or use function name
            client_id = kwargs.get('client_id', func.__name__)
            
            # Create rate limiter
            limiter = RateLimiter(max_requests, window_seconds)
            
            # Check rate limit
            if not await limiter.is_allowed(client_id):
                raise Exception(f"Rate limit exceeded for {client_id}")
            
            # Record request
            await limiter.record_request(client_id)
            
            # Execute function
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


# Context manager for rate limiting
class RateLimitContext:
    """Context manager for rate limiting"""

    def __init__(self, client_id: str, limiter: RateLimiter):
        self.client_id = client_id
        self.limiter = limiter
        self.allowed = False

    async def __aenter__(self):
        self.allowed = await self.limiter.is_allowed(self.client_id)
        if self.allowed:
            await self.limiter.record_request(self.client_id)
        return self.allowed

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass