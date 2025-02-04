"""Helper utilities for PoliticianAI."""

import json
import logging.config
import os
from pathlib import Path
from typing import Any, Dict, Optional

from src.config import LOGGING_CONFIG

def setup_logging(
    config: Optional[Dict[str, Any]] = None,
    log_file: Optional[str] = None
) -> None:
    """
    Set up logging configuration.
    
    Args:
        config: Optional custom logging configuration
        log_file: Optional log file path
    """
    try:
        # Use provided config or default
        log_config = config or LOGGING_CONFIG
        
        # Update log file path if provided
        if log_file:
            log_config["handlers"]["file"]["filename"] = log_file
        
        # Create log directory if needed
        log_path = Path(log_config["handlers"]["file"]["filename"]).parent
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.config.dictConfig(log_config)
        
    except Exception as e:
        # Fallback to basic configuration
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        logging.error(f"Error setting up logging: {str(e)}")

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
    """
    with open(config_path) as f:
        return json.load(f)

def ensure_directory(path: str) -> None:
    """
    Ensure directory exists, create if needed.
    
    Args:
        path: Directory path
    """
    Path(path).mkdir(parents=True, exist_ok=True)

def get_project_root() -> Path:
    """
    Get project root directory.
    
    Returns:
        Path to project root
    """
    return Path(__file__).parent.parent.parent

def format_error(error: Exception) -> Dict[str, str]:
    """
    Format exception for API response.
    
    Args:
        error: Exception to format
        
    Returns:
        Formatted error dictionary
    """
    return {
        "error": str(error),
        "error_type": error.__class__.__name__
    }

def truncate_text(text: str, max_length: int = 512) -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # Remove special characters
    text = text.replace("\n", " ").replace("\t", " ")
    
    return text.strip()

def get_env_bool(key: str, default: bool = False) -> bool:
    """
    Get boolean environment variable.
    
    Args:
        key: Environment variable key
        default: Default value if not set
        
    Returns:
        Boolean value
    """
    value = os.getenv(key)
    if value is None:
        return default
    return value.lower() in ("true", "1", "t", "yes", "y")

def get_env_int(key: str, default: int = 0) -> int:
    """
    Get integer environment variable.
    
    Args:
        key: Environment variable key
        default: Default value if not set
        
    Returns:
        Integer value
    """
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default

def get_env_float(key: str, default: float = 0.0) -> float:
    """
    Get float environment variable.
    
    Args:
        key: Environment variable key
        default: Default value if not set
        
    Returns:
        Float value
    """
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default
