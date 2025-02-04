import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# Database settings
DATABASE_URL = f"sqlite:///{DATA_DIR}/main.db"
CACHE_DATABASE_URL = f"sqlite:///{DATA_DIR}/cache.db"

# Model settings
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
CONTEXT_MODEL = "facebook/bart-large-mnli"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Vector store settings
EMBEDDING_DIMENSION = 768
FAISS_INDEX_PATH = DATA_DIR / "embeddings"

# Cache settings
CACHE_EXPIRY_HOURS = 24

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_WORKERS = int(os.getenv("API_WORKERS", "1"))

# LangChain settings
LANGCHAIN_VERBOSE = os.getenv("LANGCHAIN_VERBOSE", "false").lower() == "true"
LANGCHAIN_CACHE = os.getenv("LANGCHAIN_CACHE", "true").lower() == "true"

# Topics for classification
POLITICAL_TOPICS = [
    "economy",
    "healthcare",
    "immigration",
    "foreign_policy",
    "climate_change",
    "education",
    "taxes",
    "national_security",
    "social_issues",
    "infrastructure"
]

# Response templates
RESPONSE_TEMPLATES = {
    "not_found": "I don't have specific information about {topic} for {politician}.",
    "error": "I encountered an error processing your request. Please try again.",
    "clarification": "Could you please be more specific about what aspect of {topic} you'd like to know?",
    "no_data": "I don't have enough data to provide a reliable answer about that."
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "INFO"
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": "app.log",
            "formatter": "default",
            "level": "DEBUG"
        }
    },
    "root": {
        "handlers": ["console", "file"],
        "level": "INFO"
    }
}
