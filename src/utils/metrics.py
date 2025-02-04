"""Metrics collection and monitoring for the PoliticianAI project."""

import time
from functools import wraps
from typing import Dict, Optional

from prometheus_client import Counter, Gauge, Histogram, Summary

# Request metrics
HTTP_REQUEST_COUNTER = Counter(
    'http_requests_total',
    'Total number of HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency in seconds',
    ['method', 'endpoint']
)

# Model metrics
MODEL_INFERENCE_TIME = Summary(
    'model_inference_duration_seconds',
    'Time spent on model inference',
    ['model_name']
)

MODEL_LOADING_FAILURES = Counter(
    'model_loading_failures_total',
    'Number of model loading failures',
    ['model_name']
)

# Database metrics
DB_QUERY_TIME = Histogram(
    'database_query_duration_seconds',
    'Database query latency in seconds',
    ['operation']
)

DB_CONNECTION_ERRORS = Counter(
    'database_connection_errors_total',
    'Number of database connection errors'
)

# Cache metrics
CACHE_HITS = Counter('cache_hits_total', 'Number of cache hits')
CACHE_MISSES = Counter('cache_misses_total', 'Number of cache misses')
CACHE_REQUESTS = Counter('cache_requests_total', 'Total number of cache requests')

# Resource metrics
MEMORY_USAGE = Gauge('memory_usage_bytes', 'Memory usage in bytes')
GPU_MEMORY_USAGE = Gauge('gpu_memory_usage_bytes', 'GPU memory usage in bytes')
MODEL_MEMORY_USAGE = Gauge(
    'model_memory_usage_bytes',
    'Model memory usage in bytes',
    ['model_name']
)

def track_request_metrics(endpoint: str):
    """
    Decorator to track HTTP request metrics.
    
    Args:
        endpoint: API endpoint name
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                response = await func(*args, **kwargs)
                status = response.status_code
            except Exception as e:
                status = 500
                raise e
            finally:
                duration = time.time() - start_time
                HTTP_REQUEST_COUNTER.labels(
                    method='POST',
                    endpoint=endpoint,
                    status=status
                ).inc()
                REQUEST_LATENCY.labels(
                    method='POST',
                    endpoint=endpoint
                ).observe(duration)
            return response
        return wrapper
    return decorator

def track_model_inference(model_name: str):
    """
    Decorator to track model inference metrics.
    
    Args:
        model_name: Name of the model
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with MODEL_INFERENCE_TIME.labels(model_name=model_name).time():
                return func(*args, **kwargs)
        return wrapper
    return decorator

def track_database_query(operation: str):
    """
    Decorator to track database query metrics.
    
    Args:
        operation: Type of database operation
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                DB_QUERY_TIME.labels(operation=operation).observe(duration)
        return wrapper
    return decorator

def update_resource_metrics(memory_stats: Dict[str, float]):
    """
    Update resource usage metrics.
    
    Args:
        memory_stats: Dictionary containing memory usage statistics
    """
    MEMORY_USAGE.set(memory_stats['ram_used_gb'] * 1024 * 1024 * 1024)  # Convert to bytes
    if 'gpu_memory_used_gb' in memory_stats:
        GPU_MEMORY_USAGE.set(memory_stats['gpu_memory_used_gb'] * 1024 * 1024 * 1024)

def track_cache_access(hit: Optional[bool] = None):
    """
    Track cache access metrics.
    
    Args:
        hit: True if cache hit, False if miss, None if just tracking requests
    """
    CACHE_REQUESTS.inc()
    if hit is not None:
        if hit:
            CACHE_HITS.inc()
        else:
            CACHE_MISSES.inc()

def update_model_memory(model_name: str, memory_bytes: int):
    """
    Update model memory usage metrics.
    
    Args:
        model_name: Name of the model
        memory_bytes: Memory usage in bytes
    """
    MODEL_MEMORY_USAGE.labels(model_name=model_name).set(memory_bytes)
