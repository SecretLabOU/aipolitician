"""Local model configuration for the political agent.

This module handles setting up and configuring local models.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Union, List

# Global model references
models = {}

def setup_models():
    """Set up the local models needed for the political agent.
    
    This includes:
    - Mixtral 8x7B for general AI tasks
    - Fine-tuned Trump model for Trump-specific responses
    """
    global models
    
    try:
        # Use Ollama for Mixtral base model
        from langchain_community.llms import Ollama
        
        print("Setting up base Mixtral model...")
        models["mistral"] = Ollama(model="mistral")
        
        # Check for Trump model
        trump_model_path = Path("../../fine_tuned_trump_mistral/model.gguf").resolve()
        if trump_model_path.exists():
            try:
                from langchain_community.llms import LlamaCpp
                
                # Load the Trump model with appropriate parameters
                models["trump"] = LlamaCpp(
                    model_path=str(trump_model_path),
                    temperature=0.7,
                    max_tokens=512,
                    top_p=0.95,
                    verbose=False,
                    n_gpu_layers=1
                )
                print(f"Trump model loaded successfully from {trump_model_path}")
            except ImportError:
                print("Warning: llama-cpp-python not installed. Trump model will not be available.")
        else:
            # Fallback to using Mistral with specific prompt engineering
            print("Trump model not found, using Mistral with prompt engineering for Trump simulation")
            models["trump"] = models["mistral"]
            
    except ImportError:
        print("Warning: langchain_community.llms.Ollama not available, falling back to OpenAI")
        try:
            from langchain_openai import ChatOpenAI
            models["mistral"] = ChatOpenAI(model_name="gpt-3.5-turbo")
            models["trump"] = ChatOpenAI(model_name="gpt-3.5-turbo")
        except ImportError:
            print("Error: Neither Ollama nor OpenAI are available. Please install one of them.")
    
    return models

def get_model(model_name: str):
    """Get a model by name."""
    if not models:
        setup_models()
    return models.get(model_name, models.get("mistral"))