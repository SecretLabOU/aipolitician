#!/usr/bin/env python3
"""Script to download and set up required models."""

import logging
import os
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from src.config import (
    DEVICE,
    MODEL_PRECISION,
    MODELS_DIR,
    RESPONSE_MODEL
)
from src.utils import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

def get_torch_dtype(precision: str) -> torch.dtype:
    """
    Convert precision string to torch dtype.
    
    Args:
        precision: Precision string ('float16', 'float32', etc.)
        
    Returns:
        torch.dtype
    """
    dtype_map = {
        'float16': torch.float16,
        'float32': torch.float32,
        'bfloat16': torch.bfloat16
    }
    return dtype_map.get(precision.lower(), torch.float32)

def setup_model_directory():
    """Create model directory if it doesn't exist."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Model directory ready: {MODELS_DIR}")

def download_model(model_name: str):
    """
    Download and save model.
    
    Args:
        model_name: HuggingFace model name
    """
    try:
        logger.info(f"Downloading response model: {model_name}")
        
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=MODELS_DIR,
            use_fast=True
        )
        logger.info(f"Downloaded tokenizer")
        
        # Download model
        model = AutoModel.from_pretrained(
            model_name,
            cache_dir=MODELS_DIR,
            torch_dtype=get_torch_dtype(MODEL_PRECISION),
            device_map=DEVICE if torch.cuda.is_available() else None
        )
        logger.info(f"Downloaded model")
        
        # Save model and tokenizer
        model_dir = MODELS_DIR / model_name.split('/')[-1]
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        logger.info(f"Saved model and tokenizer to {model_dir}")
        
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        raise

def verify_models():
    """Verify response model is downloaded."""
    model_dir = MODELS_DIR / RESPONSE_MODEL.split('/')[-1]
    return (model_dir / "config.json").exists()

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
        
        # Check if model needs to be downloaded
        if not verify_models():
            logger.info("Downloading response model...")
            download_model(RESPONSE_MODEL)
        else:
            logger.info("Response model already downloaded")
        
        logger.info("Model setup complete!")
        
    except Exception as e:
        logger.error(f"Model setup failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
