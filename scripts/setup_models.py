#!/usr/bin/env python3
"""Script to download and set up required models."""

import logging
import os
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from src.config import (
    CONTEXT_MODEL,
    DEVICE,
    MODEL_PRECISION,
    MODELS_DIR,
    RESPONSE_MODEL,
    SENTIMENT_MODEL
)
from src.utils import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

def setup_model_directory():
    """Create model directory if it doesn't exist."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Model directory ready: {MODELS_DIR}")

def download_model(model_name: str, model_type: str):
    """
    Download and save model.
    
    Args:
        model_name: HuggingFace model name
        model_type: Type of model (for logging)
    """
    try:
        logger.info(f"Downloading {model_type} model: {model_name}")
        
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=MODELS_DIR,
            use_fast=True
        )
        logger.info(f"Downloaded {model_type} tokenizer")
        
        # Download model
        model = AutoModel.from_pretrained(
            model_name,
            cache_dir=MODELS_DIR,
            torch_dtype=MODEL_PRECISION,
            device_map=DEVICE if torch.cuda.is_available() else None
        )
        logger.info(f"Downloaded {model_type} model")
        
        # Save model and tokenizer
        model_dir = MODELS_DIR / model_name.split('/')[-1]
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        logger.info(f"Saved {model_type} model and tokenizer to {model_dir}")
        
    except Exception as e:
        logger.error(f"Error downloading {model_type} model: {str(e)}")
        raise

def verify_models():
    """Verify all required models are downloaded."""
    models = [
        (SENTIMENT_MODEL, "sentiment analysis"),
        (CONTEXT_MODEL, "context extraction"),
        (RESPONSE_MODEL, "response generation")
    ]
    
    missing_models = []
    for model_name, model_type in models:
        model_dir = MODELS_DIR / model_name.split('/')[-1]
        if not (model_dir / "config.json").exists():
            missing_models.append((model_name, model_type))
    
    return missing_models

def main():
    """Main setup function."""
    try:
        logger.info("Starting model setup...")
        
        # Create model directory
        setup_model_directory()
        
        # Check CUDA availability
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            logger.info(f"Using device: {DEVICE}")
            logger.info(f"Model precision: {MODEL_PRECISION}")
        else:
            logger.warning("CUDA not available, using CPU")
        
        # Check for missing models
        missing_models = verify_models()
        
        if missing_models:
            logger.info("Downloading missing models...")
            for model_name, model_type in missing_models:
                download_model(model_name, model_type)
        else:
            logger.info("All models already downloaded")
        
        logger.info("Model setup complete!")
        
    except Exception as e:
        logger.error(f"Model setup failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
