"""Configuration settings for PoliticianAI."""

import logging
import os
from pathlib import Path
from typing import Dict, Union

from src.config.env_utils import get_env_bool, get_env_int

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = get_env_int("API_PORT", 8000)
API_WORKERS = get_env_int("API_WORKERS", 1)

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
RESPONSE_MODEL = "gpt2"  # Base GPT-2 model for text generation

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

# API configuration
API_CONFIG = {
    "title": "PoliticianAI",
    "description": "AI system for political discourse simulation",
    "version": "1.0.0",
    "debug": DEBUG
}

__all__ = [
    'PROJECT_ROOT',
    'DATA_DIR',
    'MODELS_DIR',
    'API_HOST',
    'API_PORT',
    'API_WORKERS',
    'DEVICE',
    'MODEL_PRECISION',
    'BATCH_SIZE',
    'DEBUG',
    'LOG_LEVEL',
    'LOG_FILE',
    'RESPONSE_MODEL',
    'LOGGING_CONFIG',
    'API_CONFIG',
    'get_env_bool',
    'get_env_int'
]
