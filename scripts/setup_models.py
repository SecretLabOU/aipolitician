#!/usr/bin/env python3
"""Script to download and set up required models."""

import logging
import os
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer

from src.config import (
    MODEL_DIR,
    SENTIMENT_MODEL,
    CONTEXT_MODEL,
    EMBEDDING_MODEL,
    LOGGING_CONFIG
)

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

def setup_model_directory():
    """Create model directory if it doesn't exist."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Model directory: {MODEL_DIR}")

def download_model(model_name: str, model_type: str):
    """
    Download model from Hugging Face.
    
    Args:
        model_name: Name of the model on Hugging Face
        model_type: Type of model (for logging)
    """
    logger.info(f"Setting up {model_type} model: {model_name}")
    
    try:
        # Create model directory
        save_dir = MODEL_DIR / model_type
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Download tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # Save locally
        tokenizer.save_pretrained(save_dir)
        model.save_pretrained(save_dir)
        
        logger.info(f"Successfully downloaded and saved {model_type} model")
        
    except Exception as e:
        logger.error(f"Error downloading {model_type} model: {str(e)}")
        raise

def verify_gpu():
    """Verify GPU availability and display information."""
    logger.info("Checking GPU availability...")
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.info(f"Found {device_count} GPU(s):")
        
        for i in range(device_count):
            gpu_props = torch.cuda.get_device_properties(i)
            logger.info(f"  GPU {i}: {gpu_props.name}")
            logger.info(f"    Memory: {gpu_props.total_memory / 1024**3:.1f} GB")
            logger.info(f"    Compute capability: {gpu_props.major}.{gpu_props.minor}")
    else:
        logger.warning("No GPU found, will use CPU")
        logger.warning("Warning: Performance may be significantly reduced")

def main():
    """Main setup function."""
    try:
        logger.info("Starting model setup...")
        
        # Create model directory
        setup_model_directory()
        
        # Verify GPU
        verify_gpu()
        
        # Download models
        models_to_download = [
            (SENTIMENT_MODEL, "sentiment"),
            (CONTEXT_MODEL, "context"),
            (EMBEDDING_MODEL, "embedding")
        ]
        
        for model_name, model_type in models_to_download:
            download_model(model_name, model_type)
        
        logger.info("Model setup complete!")
        
        # Verify setup
        missing_models = []
        required_models = ["sentiment", "context", "embedding"]
        
        for model in required_models:
            if not (MODEL_DIR / model).exists():
                missing_models.append(model)
        
        if missing_models:
            logger.warning("The following models are missing:")
            for model in missing_models:
                logger.warning(f"  - {model}")
            logger.warning("Please ensure all models are properly downloaded")
            sys.exit(1)
        else:
            logger.info("All required models are present")
        
    except KeyboardInterrupt:
        logger.error("Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during setup: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
