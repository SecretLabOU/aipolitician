"""Configuration for local models.

This module provides configuration and management for local LLM models
used in the political agent graph system.
"""

import json
import os
from typing import Dict, Any, List

# Define paths
DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), 
    "../../models/config.json"
)

# Default configuration for local models
DEFAULT_MODEL_CONFIG = {
    "models": {
        "llama3": {
            "type": "llamacpp",
            "path": "models/llama-3-8b-instruct.Q4_K_M.gguf",
            "n_gpu_layers": 1,
            "n_ctx": 4096,
            "temperature": 0.7,
            "description": "Llama 3 8B instruct model (quantized)"
        },
        "mistral": {
            "type": "ollama",
            "model": "mistral",
            "temperature": 0.7,
            "description": "Mistral 7B instruct model via Ollama"
        },
        "mixtral": {
            "type": "ollama",
            "model": "mixtral",
            "temperature": 0.7,
            "description": "Mixtral 8x7B instruct model via Ollama"
        },
        "trump_mistral": {
            "type": "llamacpp",
            "path": "../../fine_tuned_trump_mistral/model.gguf",
            "n_gpu_layers": 1,
            "n_ctx": 4096,
            "temperature": 0.8,
            "description": "Fine-tuned Mistral Trump model"
        }
    },
    "default_model": "trump_mistral",
    "model_folder": "models"
}

def save_default_config():
    """Save the default configuration file if it doesn't exist."""
    config_dir = os.path.dirname(DEFAULT_CONFIG_PATH)
    
    # Create the models directory if it doesn't exist
    if not os.path.exists(config_dir):
        os.makedirs(config_dir, exist_ok=True)
    
    # Save the default configuration if it doesn't exist
    if not os.path.exists(DEFAULT_CONFIG_PATH):
        with open(DEFAULT_CONFIG_PATH, 'w') as f:
            json.dump(DEFAULT_MODEL_CONFIG, f, indent=2)
        print(f"Created default model configuration at {DEFAULT_CONFIG_PATH}")

def load_model_config() -> Dict[str, Any]:
    """Load the model configuration from the config file."""
    # Ensure default config exists
    save_default_config()
    
    try:
        with open(DEFAULT_CONFIG_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading model config: {e}. Using default configuration.")
        return DEFAULT_MODEL_CONFIG

def list_available_models() -> List[Dict[str, str]]:
    """List all available local models with their descriptions."""
    config = load_model_config()
    
    return [
        {
            "name": name,
            "type": details["type"],
            "description": details.get("description", "No description")
        }
        for name, details in config["models"].items()
    ]

def get_model_details(model_name: str) -> Dict[str, Any]:
    """Get details for a specific model."""
    config = load_model_config()
    
    if model_name not in config["models"]:
        raise ValueError(f"Unknown model: {model_name}")
    
    return config["models"][model_name]

def get_default_model() -> str:
    """Get the name of the default model."""
    config = load_model_config()
    return config.get("default_model", "mixtral")

# Initialize the default configuration
save_default_config()