"""Utility functions for the PoliticianAI project."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from sqlalchemy.orm import Session

from src.config import LOGGING_CONFIG

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

def setup_gpu() -> torch.device:
    """
    Set up GPU if available, otherwise use CPU.
    
    Returns:
        torch.device: Device to use for computations
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("GPU not available, using CPU")
    return device

def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage statistics.
    
    Returns:
        Dict containing memory usage information
    """
    memory_stats = {
        "cpu_percent": 0.0,
        "ram_used_gb": 0.0,
        "gpu_memory_used_gb": 0.0
    }
    
    try:
        import psutil
        process = psutil.Process()
        memory_stats["cpu_percent"] = process.cpu_percent()
        memory_stats["ram_used_gb"] = process.memory_info().rss / (1024 ** 3)
        
        if torch.cuda.is_available():
            memory_stats["gpu_memory_used_gb"] = (
                torch.cuda.memory_allocated() / (1024 ** 3)
            )
    except Exception as e:
        logger.warning(f"Error getting memory stats: {str(e)}")
    
    return memory_stats

def safe_db_commit(session: Session, error_msg: str = "Database error") -> bool:
    """
    Safely commit database changes with error handling.
    
    Args:
        session: SQLAlchemy session
        error_msg: Custom error message
        
    Returns:
        bool: True if commit successful, False otherwise
    """
    try:
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        logger.error(f"{error_msg}: {str(e)}")
        return False

def create_directory(path: Union[str, Path]) -> bool:
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Directory path to create
        
    Returns:
        bool: True if directory exists or was created
    """
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error creating directory {path}: {str(e)}")
        return False

def format_timestamp(dt: Optional[datetime] = None) -> str:
    """
    Format timestamp for logging and display.
    
    Args:
        dt: Datetime object (uses current time if None)
        
    Returns:
        str: Formatted timestamp
    """
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def batch_process(items: List[Any], batch_size: int) -> List[List[Any]]:
    """
    Split list into batches for processing.
    
    Args:
        items: List of items to batch
        batch_size: Size of each batch
        
    Returns:
        List of batches
    """
    return [
        items[i:i + batch_size]
        for i in range(0, len(items), batch_size)
    ]

def log_execution_time(func: callable) -> callable:
    """
    Decorator to log function execution time.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function
    """
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        execution_time = (datetime.now() - start_time).total_seconds()
        logger.debug(f"{func.__name__} executed in {execution_time:.2f} seconds")
        return result
    return wrapper

def sanitize_input(text: str) -> str:
    """
    Sanitize user input for safety.
    
    Args:
        text: Input text to sanitize
        
    Returns:
        str: Sanitized text
    """
    # Remove any potentially harmful characters
    text = ''.join(char for char in text if char.isprintable())
    # Limit length
    return text[:1000]  # Arbitrary limit for safety
