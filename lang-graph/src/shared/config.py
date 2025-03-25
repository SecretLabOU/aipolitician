"""
System configuration for Political Agent.

Centralized configuration with automatic hardware detection.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
import torch
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Default paths
ROOT_DIR = Path(__file__).parent.parent.parent.parent
MODELS_DIR = ROOT_DIR / "models"
CONFIG_PATH = MODELS_DIR / "config.json"
DATA_DIR = ROOT_DIR / "data"

# Hardware detection
HARDWARE_CONFIG = {
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    "device_names": [],
    "vram_gb": [],
    "detected_preset": "cpu"
}

if HARDWARE_CONFIG["cuda_available"]:
    for i in range(HARDWARE_CONFIG["device_count"]):
        name = torch.cuda.get_device_name(i)
        HARDWARE_CONFIG["device_names"].append(name)
        vram = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        HARDWARE_CONFIG["vram_gb"].append(vram)
        
        # Auto-detect preset based on GPU
        if "4090" in name:
            HARDWARE_CONFIG["detected_preset"] = "rtx4090"
        elif "4060" in name and "Ti" in name:
            HARDWARE_CONFIG["detected_preset"] = "rtx4060ti"
        elif not HARDWARE_CONFIG["detected_preset"] == "rtx4090":
            HARDWARE_CONFIG["detected_preset"] = "gpu"  # Generic GPU

# Log hardware detection results
if HARDWARE_CONFIG["cuda_available"]:
    logger.info(f"Detected {HARDWARE_CONFIG['device_count']} CUDA devices:")
    for i in range(HARDWARE_CONFIG["device_count"]):
        logger.info(f"  Device {i}: {HARDWARE_CONFIG['device_names'][i]} with {HARDWARE_CONFIG['vram_gb'][i]:.1f}GB VRAM")
    logger.info(f"Using preset: {HARDWARE_CONFIG['detected_preset']}")
else:
    logger.info("No CUDA devices detected. Using CPU mode.")

# GPU optimizations based on detected hardware
GPU_OPTIMIZATIONS = {
    "rtx4090": {
        "n_ctx": 16384,
        "n_batch": 2048,
        "tensor_split": None,  # Will be auto-configured if multiple GPUs
        "use_flash_attn": True,
        "rope_scaling_type": "linear",
        "parallel_processes": 4
    },
    "rtx4060ti": {
        "n_ctx": 8192,
        "n_batch": 1024,
        "tensor_split": None,
        "use_flash_attn": True,
        "parallel_processes": 3
    },
    "gpu": {
        "n_ctx": 4096,
        "n_batch": 512,
        "tensor_split": None,
        "use_flash_attn": False,
        "parallel_processes": 2
    },
    "cpu": {
        "n_ctx": 2048,
        "n_batch": 256,
        "tensor_split": None,
        "use_flash_attn": False,
        "parallel_processes": 1
    }
}

# Cache for loaded configuration
_config_cache = None

def load_config() -> Dict[str, Any]:
    """Load configuration from config.json with caching."""
    global _config_cache
    
    if _config_cache is not None:
        return _config_cache
    
    # Create models directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Load config if it exists, else create a default config
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {CONFIG_PATH}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            config = create_default_config()
    else:
        logger.warning(f"Config file not found at {CONFIG_PATH}, creating default")
        config = create_default_config()
    
    # Add system configuration
    config["_system"] = {
        "hardware": HARDWARE_CONFIG,
        "optimizations": GPU_OPTIMIZATIONS[HARDWARE_CONFIG["detected_preset"]]
    }
    
    # Cache for future use
    _config_cache = config
    return config

def create_default_config() -> Dict[str, Any]:
    """Create a default configuration."""
    config = {
        "trump": {
            "model_path": "models/trump-7b.gguf",
            "display_name": "Donald Trump",
            "party": "Republican"
        },
        "biden": {
            "model_path": "models/biden-7b.gguf",
            "display_name": "Joe Biden",
            "party": "Democrat"
        }
    }
    
    # Save the default config
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Created default configuration at {CONFIG_PATH}")
    except Exception as e:
        logger.error(f"Error saving default config: {e}")
    
    return config

def get_optimization_config(preset: Optional[str] = None) -> Dict[str, Any]:
    """Get GPU optimization settings for a specific preset."""
    if preset is None:
        preset = HARDWARE_CONFIG["detected_preset"]
    
    return GPU_OPTIMIZATIONS.get(preset, GPU_OPTIMIZATIONS["cpu"])

def get_model_path(persona_id: str) -> Optional[str]:
    """Get the model path for a specific persona."""
    config = load_config()
    
    if persona_id in config:
        model_path = config[persona_id].get("model_path")
        if model_path:
            # Convert to absolute path if necessary
            if not os.path.isabs(model_path):
                model_path = str(ROOT_DIR / model_path)
            return model_path
    
    return None

def get_system_info() -> Dict[str, Any]:
    """Get system information including hardware details."""
    config = load_config()
    return config.get("_system", {}) 