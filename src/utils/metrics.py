"""Metrics utilities for PoliticianAI."""

import logging
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from sqlalchemy.orm import Session

from src.utils import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

# Metrics storage
_metrics: Dict[str, Dict[str, Union[int, float, List[float]]]] = {
    "requests": {
        "total": 0,
        "success": 0,
        "error": 0,
        "latencies": []
    },
    "models": {
        "inference_count": 0,
        "inference_time": 0.0,
        "cache_hits": 0,
        "cache_misses": 0
    },
    "database": {
        "queries": 0,
        "errors": 0,
        "latencies": []
    }
}

def track_request(success: bool = True, latency: Optional[float] = None) -> None:
    """
    Track API request metrics.
    
    Args:
        success: Whether request was successful
        latency: Request latency in seconds
    """
    _metrics["requests"]["total"] += 1
    if success:
        _metrics["requests"]["success"] += 1
    else:
        _metrics["requests"]["error"] += 1
    
    if latency is not None:
        _metrics["requests"]["latencies"].append(latency)

def track_model_inference(duration: float) -> None:
    """
    Track model inference metrics.
    
    Args:
        duration: Inference duration in seconds
    """
    _metrics["models"]["inference_count"] += 1
    _metrics["models"]["inference_time"] += duration

def track_cache_access(hit: bool) -> None:
    """
    Track cache access metrics.
    
    Args:
        hit: Whether access was a cache hit
    """
    if hit:
        _metrics["models"]["cache_hits"] += 1
    else:
        _metrics["models"]["cache_misses"] += 1

def track_database_query(success: bool = True, latency: Optional[float] = None) -> None:
    """
    Track database query metrics.
    
    Args:
        success: Whether query was successful
        latency: Query latency in seconds
    """
    _metrics["database"]["queries"] += 1
    if not success:
        _metrics["database"]["errors"] += 1
    
    if latency is not None:
        _metrics["database"]["latencies"].append(latency)

def calculate_metrics() -> Dict[str, Any]:
    """
    Calculate current metrics.
    
    Returns:
        Dictionary containing:
            - request_count: Total number of requests
            - success_rate: Request success rate
            - avg_latency: Average request latency
            - model_inferences: Number of model inferences
            - avg_inference_time: Average model inference time
            - cache_hit_rate: Cache hit rate
            - database_queries: Number of database queries
            - database_error_rate: Database error rate
            - avg_query_latency: Average database query latency
    """
    # Calculate request metrics
    total_requests = _metrics["requests"]["total"]
    success_rate = (
        _metrics["requests"]["success"] / total_requests
        if total_requests > 0 else 0.0
    )
    avg_latency = (
        sum(_metrics["requests"]["latencies"]) / len(_metrics["requests"]["latencies"])
        if _metrics["requests"]["latencies"] else 0.0
    )
    
    # Calculate model metrics
    inference_count = _metrics["models"]["inference_count"]
    avg_inference_time = (
        _metrics["models"]["inference_time"] / inference_count
        if inference_count > 0 else 0.0
    )
    cache_hits = _metrics["models"]["cache_hits"]
    cache_misses = _metrics["models"]["cache_misses"]
    cache_hit_rate = (
        cache_hits / (cache_hits + cache_misses)
        if cache_hits + cache_misses > 0 else 0.0
    )
    
    # Calculate database metrics
    total_queries = _metrics["database"]["queries"]
    error_rate = (
        _metrics["database"]["errors"] / total_queries
        if total_queries > 0 else 0.0
    )
    avg_query_latency = (
        sum(_metrics["database"]["latencies"]) / len(_metrics["database"]["latencies"])
        if _metrics["database"]["latencies"] else 0.0
    )
    
    return {
        "request_count": total_requests,
        "success_rate": success_rate,
        "avg_latency": avg_latency,
        "model_inferences": inference_count,
        "avg_inference_time": avg_inference_time,
        "cache_hit_rate": cache_hit_rate,
        "database_queries": total_queries,
        "database_error_rate": error_rate,
        "avg_query_latency": avg_query_latency
    }

def track_performance(metric_type: str) -> Callable:
    """
    Decorator to track function performance metrics.
    
    Args:
        metric_type: Type of metric to track ("request", "model", or "database")
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            success = True
            try:
                result = func(*args, **kwargs)
                return result
            except Exception:
                success = False
                raise
            finally:
                duration = time.time() - start_time
                if metric_type == "request":
                    track_request(success=success, latency=duration)
                elif metric_type == "model":
                    track_model_inference(duration=duration)
                elif metric_type == "database":
                    track_database_query(success=success, latency=duration)
        return wrapper
    return decorator
