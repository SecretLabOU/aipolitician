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

# LangChain settings
LANGCHAIN_VERBOSE = get_env_bool("LANGCHAIN_VERBOSE", False)
LANGCHAIN_CACHE = get_env_bool("LANGCHAIN_CACHE", True)

# Cache settings
CACHE_EXPIRY_HOURS = get_env_int("CACHE_EXPIRY_HOURS", 24)
RESPONSE_CACHE_SIZE = get_env_int("RESPONSE_CACHE_SIZE", 1000)

# Vector store settings
EMBEDDING_DIMENSION = get_env_int("EMBEDDING_DIMENSION", 768)
MAX_CONTEXT_LENGTH = get_env_int("MAX_CONTEXT_LENGTH", 2048)

# GPU settings
CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "0")
GPU_MEMORY_FRACTION = get_env_float("GPU_MEMORY_FRACTION", 0.9)

# Development settings
DEBUG = get_env_bool("DEBUG", False)
RELOAD = get_env_bool("RELOAD", False)
TESTING = get_env_bool("TESTING", False)

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "app.log")

# Model paths
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
CONTEXT_MODEL = "facebook/bart-large-mnli"
RESPONSE_MODEL = "microsoft/DialoGPT-large"  # Dialogue-optimized GPT model

# Political topics
POLITICAL_TOPICS: List[str] = [
    "Healthcare",
    "Economy",
    "Education",
    "Immigration",
    "Climate Change",
    "National Security",
    "Gun Control",
    "Social Security",
    "Tax Policy",
    "Foreign Policy",
    "Criminal Justice",
    "Infrastructure",
    "Energy Policy",
    "Trade Policy",
    "Civil Rights",
]

# Logging configuration
LOGGING_CONFIG: Dict[str, Union[str, Dict]] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
        "json": {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "fmt": "%(asctime)s %(levelname)s %(name)s %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": LOG_LEVEL,
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.FileHandler",
            "level": LOG_LEVEL,
            "formatter": "json",
            "filename": LOG_FILE,
            "mode": "a"
        }
    },
    "loggers": {
        "": {  # Root logger
            "handlers": ["console", "file"],
            "level": LOG_LEVEL,
            "propagate": True
        }
    }
}

# Prometheus metrics configuration
METRICS_CONFIG = {
    "namespace": "politician_ai",
    "subsystem": "",
    "enable_default_metrics": True,
    "buckets": (
        .005, .01, .025, .05, .075, .1, .25, .5, .75, 1.0, 2.5, 5.0, 7.5, 10.0
    ),
    "multiprocess_mode": "all"
}

# OpenTelemetry configuration
OTEL_CONFIG = {
    "service_name": "politician_ai",
    "environment": "production" if not DEBUG else "development",
    "sampler": "always_on" if DEBUG else "parentbased_traceidratio",
    "sampling_ratio": 1.0 if DEBUG else 0.1
}

# Cache configuration
CACHE_CONFIG = {
    "ttl": CACHE_EXPIRY_HOURS * 3600,  # Convert hours to seconds
    "maxsize": RESPONSE_CACHE_SIZE,
    "typed": False
}

# Model configuration
MODEL_CONFIG = {
    "device": DEVICE,
    "torch_dtype": MODEL_PRECISION,
    "max_length": MAX_CONTEXT_LENGTH,
    "num_beams": 4,
    "length_penalty": 2.0,
    "no_repeat_ngram_size": 3,
    "batch_size": BATCH_SIZE
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
    "docs_url": "/docs" if DEBUG else None,
    "redoc_url": "/redoc" if DEBUG else None,
    "openapi_url": "/openapi.json" if DEBUG else None,
    "root_path": "",
    "debug": DEBUG
}
