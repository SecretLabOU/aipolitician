#!/usr/bin/env python3
"""
Configuration settings for the LangGraph-based AI Politician system.
"""
import os
from dotenv import load_dotenv
from enum import Enum
from pathlib import Path

# Load environment variables
load_dotenv()

# Base paths
ROOT_DIR = Path(__file__).parent.parent.parent.parent.absolute()

# Model configurations
class PoliticianIdentity(str, Enum):
    BIDEN = "biden"
    TRUMP = "trump"

# LLM Configurations
DEFAULT_MODEL = "gpt-3.5-turbo"
if os.environ.get("OPENAI_API_KEY"):
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
else:
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Sentiment analysis thresholds
SENTIMENT_DEFLECTION_THRESHOLD = 0.3  # Sentiment score below which deflection is triggered

# RAG Configurations
ENABLE_RAG = True
try:
    from src.data.db.utils.rag_utils import integrate_with_chat
    HAS_RAG = True
except ImportError:
    HAS_RAG = False
    print("RAG database system not available. Running with synthetic responses.")

# Paths to fine-tuned models
BIDEN_ADAPTER_PATH = os.environ.get("BIDEN_ADAPTER_PATH", "nnat03/biden-mistral-adapter")
TRUMP_ADAPTER_PATH = os.environ.get("TRUMP_ADAPTER_PATH", "nnat03/trump-mistral-adapter")

# Mistral base model
BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

# Response generation parameters
MAX_RESPONSE_LENGTH = 512
DEFAULT_TEMPERATURE = 0.7 