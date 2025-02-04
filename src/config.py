"""Configuration settings for the PoliticianAI project."""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

# Model configurations
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
CONTEXT_MODEL = "facebook/bart-large"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Environment variables with defaults
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_WORKERS = int(os.getenv("API_WORKERS", "1"))

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR}/main.db")
CACHE_DATABASE_URL = os.getenv("CACHE_DATABASE_URL", f"sqlite:///{DATA_DIR}/cache.db")

# Model settings
DEVICE = os.getenv("DEVICE", "cuda")  # or "cpu"
MODEL_PRECISION = os.getenv("MODEL_PRECISION", "float16")  # or "float32"
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))

# LangChain settings
LANGCHAIN_VERBOSE = os.getenv("LANGCHAIN_VERBOSE", "false").lower() == "true"
LANGCHAIN_CACHE = os.getenv("LANGCHAIN_CACHE", "true").lower() == "true"

# Cache settings
CACHE_EXPIRY_HOURS = int(os.getenv("CACHE_EXPIRY_HOURS", "24"))
RESPONSE_CACHE_SIZE = int(os.getenv("RESPONSE_CACHE_SIZE", "1000"))

# Vector store settings
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))
MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", "2048"))

# GPU settings
CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "0")
GPU_MEMORY_FRACTION = float(os.getenv("GPU_MEMORY_FRACTION", "0.9"))

# Development settings
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
RELOAD = os.getenv("RELOAD", "false").lower() == "true"
TESTING = os.getenv("TESTING", "false").lower() == "true"

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "app.log")

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": LOG_LEVEL,
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "level": LOG_LEVEL,
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": LOG_FILE,
            "mode": "a",
        },
    },
    "loggers": {
        "": {  # root logger
            "handlers": ["default", "file"],
            "level": LOG_LEVEL,
            "propagate": True
        },
    }
}

# Available political topics
POLITICAL_TOPICS = [
    "healthcare",
    "economy",
    "climate_change",
    "education",
    "immigration",
    "foreign_policy",
    "national_security",
    "taxes",
    "social_security",
    "infrastructure",
]
