import os
from typing import Dict, Any
from dotenv import load_dotenv
import torch
from .config import MODELS, MODEL_TYPE_CONFIGS

# Load environment variables
load_dotenv()

def get_api_key() -> str:
    """Get HuggingFace API key from environment"""
    key = os.getenv("HUGGINGFACE_API_KEY")
    if not key:
        raise ValueError("HUGGINGFACE_API_KEY not found in environment variables")
    return key

def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get complete model configuration including environment-specific settings"""
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not found in configuration")
    
    # Get base config
    config = MODELS[model_name].copy()
    
    # Add model type specific configurations
    model_type = config.get("model_type")
    if model_type in MODEL_TYPE_CONFIGS:
        type_config = MODEL_TYPE_CONFIGS[model_type].copy()
        
        # Convert string dtype to torch dtype
        if "torch_dtype" in type_config:
            if type_config["torch_dtype"] == "float16":
                type_config["torch_dtype"] = torch.float16
            elif type_config["torch_dtype"] == "bfloat16":
                type_config["torch_dtype"] = torch.bfloat16
        
        config.update(type_config)
    
    return config

def validate_model_path(model_name: str) -> bool:
    """Validate that model files exist at specified path"""
    config = MODELS.get(model_name)
    if not config:
        return False
    
    lora_path = config.get("lora_weights")
    if not lora_path or not os.path.exists(lora_path):
        return False
    
    return True
