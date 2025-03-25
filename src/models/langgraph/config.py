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

# LLM Configurations - Using Open Source Models

# Sentiment Analysis Model - Lightweight option by default
SENTIMENT_MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"  # Simple, fast sentiment analysis
# Alternative sentiment models if cardiffnlp fails:
# - "SamLowe/roberta-base-go_emotions" (more detailed but heavier)
# - "finiteautomata/bertweet-base-sentiment-analysis" (good for Twitter-style text)

# Context Extraction LLM - Using smaller model by default for compatibility
CONTEXT_LLM_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Small, fast model for context extraction
# Alternative options for more powerful machines:
# - "mistralai/Mixtral-8x7B-Instruct-v0.1" (very powerful, needs 24GB+ VRAM)
# - "MBZUAI/LaMini-Flan-T5-783M" (even smaller option)

USE_4BIT_QUANTIZATION = True  # Set to True to reduce VRAM usage

# Flag for using OpenAI (now disabled by default)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
HAS_OPENAI = False  # Explicitly disabled regardless of API key presence

# Sentiment analysis thresholds
SENTIMENT_DEFLECTION_THRESHOLD = -0.3  # Sentiment score below which deflection is triggered

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