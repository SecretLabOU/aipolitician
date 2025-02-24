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
    try:
        config = MODELS.get(model_name)
        if not config:
            raise ValueError(f"Model {model_name} not found in configuration")
        
        # Check if SHARED_MODELS_PATH is set and exists
        shared_path = os.getenv("SHARED_MODELS_PATH")
        if not shared_path:
            raise ValueError("SHARED_MODELS_PATH environment variable not set")
        if not os.path.exists(shared_path):
            raise ValueError(f"SHARED_MODELS_PATH directory does not exist: {shared_path}")
            
        # Validate LoRA weights path
        lora_path = config.get("lora_weights")
        if not lora_path:
            raise ValueError(f"LoRA weights path not configured for model {model_name}")
        if not os.path.exists(lora_path):
            raise ValueError(f"LoRA weights directory not found at: {lora_path}")
            
        # Check for required adapter files
        required_files = [
            'adapter_config.json',
            'adapter_model.safetensors',  # Updated to look for safetensors instead of .bin
            'tokenizer.json',
            'tokenizer_config.json',
            'special_tokens_map.json'
        ]
        for file in required_files:
            file_path = os.path.join(lora_path, file)
            if not os.path.exists(file_path):
                raise ValueError(f"Required adapter file not found: {file_path}")
                
        return True
        
    except Exception as e:
        import logging
        logging.error(f"Model validation failed: {str(e)}")
        return False
