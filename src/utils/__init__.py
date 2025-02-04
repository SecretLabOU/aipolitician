"""Utility functions for PoliticianAI."""

import logging.config
from pathlib import Path

def setup_logging(config: dict = None, log_file: str = None) -> None:
    """
    Set up logging configuration.
    
    Args:
        config: Optional custom logging configuration
        log_file: Optional log file path
    """
    try:
        # Use provided config or import default
        if config is None:
            from src.config import LOGGING_CONFIG
            log_config = LOGGING_CONFIG
        else:
            log_config = config
        
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

# Import cache utilities after setup_logging to avoid circular imports
from .cache import Cache, CacheManager
