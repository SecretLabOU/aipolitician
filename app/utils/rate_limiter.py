from typing import Dict, Tuple
import time
from fastapi import Request, HTTPException
from .exceptions import RateLimitError

class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.window_size = 60  # 1 minute in seconds
        self._requests: Dict[str, list] = {}  # IP -> list of timestamps
        self._cleanup_interval = 300  # Cleanup every 5 minutes

    def _cleanup_old_requests(self, current_time: float):
        """Remove requests older than the window size."""
        for ip in list(self._requests.keys()):
            # Keep only requests within the window
            self._requests[ip] = [
                ts for ts in self._requests[ip]
                if current_time - ts < self.window_size
            ]
            # Remove IP if no requests remain
            if not self._requests[ip]:
                del self._requests[ip]

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP from request headers or connection info."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0]
        return request.client.host if request.client else "unknown"

    async def check_rate_limit(self, request: Request):
        """Check if request is within rate limits."""
        current_time = time.time()
        
        # Periodic cleanup of old requests
        if current_time % self._cleanup_interval < 1:
            self._cleanup_old_requests(current_time)
        
        # Get client IP
        client_ip = self._get_client_ip(request)
        
        # Initialize request list for new IPs
        if client_ip not in self._requests:
            self._requests[client_ip] = []
        
        # Remove old requests for this IP
        self._requests[client_ip] = [
            ts for ts in self._requests[client_ip]
            if current_time - ts < self.window_size
        ]
        
        # Check rate limit
        if len(self._requests[client_ip]) >= self.requests_per_minute:
            oldest_request = min(self._requests[client_ip])
            wait_time = self.window_size - (current_time - oldest_request)
            raise RateLimitError(f"Rate limit exceeded. Please wait {int(wait_time)} seconds.")
        
        # Add current request
        self._requests[client_ip].append(current_time)

    def get_rate_limit_headers(self, request: Request) -> Dict[str, str]:
        """Get rate limit headers for response."""
        client_ip = self._get_client_ip(request)
        if client_ip in self._requests:
            current_requests = len(self._requests[client_ip])
            remaining = max(0, self.requests_per_minute - current_requests)
            return {
                "X-RateLimit-Limit": str(self.requests_per_minute),
                "X-RateLimit-Remaining": str(remaining),
                "X-RateLimit-Reset": str(int(time.time() + self.window_size))
            }
        return {
            "X-RateLimit-Limit": str(self.requests_per_minute),
            "X-RateLimit-Remaining": str(self.requests_per_minute),
            "X-RateLimit-Reset": str(int(time.time() + self.window_size))
        }

# Create global rate limiter instance
rate_limiter = RateLimiter()

# FastAPI dependency for rate limiting
async def check_rate_limit(request: Request):
    """FastAPI dependency for rate limiting."""
    try:
        await rate_limiter.check_rate_limit(request)
    except RateLimitError as e:
        raise HTTPException(
            status_code=429,
            detail=str(e),
            headers=rate_limiter.get_rate_limit_headers(request)
        )
    return rate_limiter.get_rate_limit_headers(request)
