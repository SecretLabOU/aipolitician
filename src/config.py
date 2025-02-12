"""Configuration settings for PoliticianAI."""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from src.utils.helpers import get_env_bool, get_env_float, get_env_int

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = get_env_int("API_PORT", 8000)
API_WORKERS = get_env_int("API_WORKERS", 1)

# Database settings
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")
CACHE_DATABASE_URL = os.getenv("CACHE_DATABASE_URL", DATABASE_URL)

# Model settings
DEVICE = os.getenv("DEVICE", "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu")
MODEL_PRECISION = os.getenv("MODEL_PRECISION", "float16")
BATCH_SIZE = get_env_int("BATCH_SIZE", 1)

# Development settings
DEBUG = get_env_bool("DEBUG", False)

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "app.log")

# Model paths
RESPONSE_MODEL = "microsoft/DialoGPT-large"  # Professional dialogue generation model

# Logging configuration
LOGGING_CONFIG: Dict[str, Union[str, Dict]] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": LOG_LEVEL,
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        }
    },
    "loggers": {
        "": {  # Root logger
            "handlers": ["console"],
            "level": LOG_LEVEL,
            "propagate": True
        }
    }
}

# Database configuration
DB_CONFIG = {
    "pool_size": 5,
    "max_overflow": 10,
    "pool_timeout": 30,
    "pool_recycle": 1800,
    "echo": DEBUG
}

# API configuration
API_CONFIG = {
    "title": "PoliticianAI",
    "description": "AI system for political discourse simulation",
    "version": "1.0.0",
    "debug": DEBUG
}
