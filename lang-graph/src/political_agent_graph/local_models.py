"""
GPU-accelerated model implementation for political agent.

Provides advanced multi-GPU inference with tensor parallelism optimization.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any, List, Mapping, Union, Tuple
import torch
from langchain.llms.base import LLM
from llama_cpp import Llama
import threading
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Global model cache with thread-safe access
_model_cache = {}
_cache_lock = threading.RLock()

# GPU allocation strategy
_gpu_allocation = {}
_next_gpu_id = 0

class AdvancedGPUModel(LLM):
    """Advanced GPU-accelerated LLM using tensor parallelism when available."""
    
    model_path: str
    model: Optional[Llama] = None
    persona: str
    gpu_id: Optional[int] = None
    n_ctx: int = 8192  # Default to larger context for RTX 4090
    n_batch: int = 1024
    n_threads: int = None
    tensor_split: Optional[List[float]] = None
    
    def __init__(
        self, 
        model_path: str, 
        persona: str,
        gpu_id: Optional[int] = None,
        preset: Optional[str] = None
    ):
        """Initialize with optimal GPU configuration."""
        super().__init__(model_path=model_path, persona=persona)
        
        self.n_threads = os.cpu_count() or 8
        
        # Skip if model doesn't exist
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}")
            self.model = None
            return
            
        try:
            # GPU detection and allocation
            self.gpu_id = self._allocate_gpu(gpu_id)
            
            # Load optimized configuration
            gpu_params = self._get_optimal_config(preset)
            
            # Configure tensor parallelism if multiple GPUs
            self.tensor_split = self._configure_tensor_split()
            
            # Update settings from optimal config
            self.n_ctx = gpu_params.get("n_ctx", self.n_ctx)
            self.n_batch = gpu_params.get("n_batch", self.n_batch)
            
            # Log GPU configuration
            logger.info(f"Loading {persona} model on GPU {self.gpu_id if self.gpu_id is not None else 'CPU'}")
            if self.tensor_split:
                logger.info(f"Using tensor parallelism with split: {self.tensor_split}")
            
            # Load the model with optimized settings
            self.model = Llama(
                model_path=model_path,
                n_ctx=self.n_ctx,
                n_batch=self.n_batch,
                n_gpu_layers=-1,  # All layers on GPU
                n_threads=self.n_threads,
                tensor_split=self.tensor_split,
                verbose=False
            )
            
            logger.info(f"Model loaded successfully with context size {self.n_ctx}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
    
    def _allocate_gpu(self, requested_gpu: Optional[int] = None) -> Optional[int]:
        """Allocate GPU based on availability and current load."""
        global _next_gpu_id, _gpu_allocation
        
        # If CPU requested or no CUDA, use CPU
        if requested_gpu == -1 or not torch.cuda.is_available():
            return None
        
        # Return specific GPU if requested
        if requested_gpu is not None and requested_gpu < torch.cuda.device_count():
            return requested_gpu
        
        with _cache_lock:
            # Auto allocation based on VRAM availability
            if torch.cuda.device_count() > 1:
                # Find GPU with most free memory
                free_memory = []
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        total = torch.cuda.get_device_properties(i).total_memory
                        reserved = torch.cuda.memory_reserved(i)
                        allocated = torch.cuda.memory_allocated(i)
                        free = total - reserved - allocated
                        free_memory.append((i, free))
                
                # Sort by free memory (descending)
                free_memory.sort(key=lambda x: x[1], reverse=True)
                return free_memory[0][0]
            elif torch.cuda.device_count() == 1:
                return 0
        
        return None
    
    def _get_optimal_config(self, preset: Optional[str] = None) -> Dict[str, Any]:
        """Get optimal configuration based on GPU."""
        if preset == "rtx4090":
            return {
                "n_ctx": 16384,  # Extended context for 4090
                "n_batch": 2048,
                "kv_overrides": {
                    "use_flash_attn": True,
                    "rope_scaling_type": "linear"
                }
            }
        elif preset == "rtx4060ti":
            return {
                "n_ctx":
                 8192,  # Good balance for 4060Ti
                "n_batch": 1024,
                "kv_overrides": {
                    "use_flash_attn": True
                }
            }
        
        # Auto-detect based on VRAM size
        config = {
            "n_ctx": 4096,
            "n_batch": 512
        }
        
        try:
            if self.gpu_id is not None:
                props = torch.cuda.get_device_properties(self.gpu_id)
                memory_gb = props.total_memory / (1024 ** 3)
                
                # Scale context based on available VRAM
                if memory_gb >= 22:  # RTX 4090
                    config["n_ctx"] = 16384
                    config["n_batch"] = 2048
                elif memory_gb >= 15:  # RTX 4060 Ti
                    config["n_ctx"] = 8192
                    config["n_batch"] = 1024
                elif memory_gb >= 8:
                    config["n_ctx"] = 4096
                    config["n_batch"] = 512
        except Exception as e:
            logger.warning(f"Error determining GPU config: {e}")
        
        return config
    
    def _configure_tensor_split(self) -> Optional[List[float]]:
        """Configure tensor split for multi-GPU inference."""
        if torch.cuda.device_count() <= 1:
            return None
            
        # For simplicity we'll use the first 2 GPUs if available
        # You could extend this for more GPUs with more sophisticated allocation
        if torch.cuda.device_count() >= 2:
            # Check VRAM capacity for both GPUs
            vram_0 = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            vram_1 = torch.cuda.get_device_properties(1).total_memory / (1024**3)
            
            # If one GPU is significantly larger (like RTX 4090 vs 4060 Ti)
            if vram_0 >= 1.5 * vram_1:
                # Allocate proportionally more to the larger GPU
                total = vram_0 + vram_1
                split_0 = vram_0 / total
                split_1 = vram_1 / total
                return [split_0, split_1]
            elif vram_1 >= 1.5 * vram_0:
                # Larger second GPU
                total = vram_0 + vram_1
                split_0 = vram_0 / total
                split_1 = vram_1 / total
                return [split_0, split_1]
            else:
                # Similar sized GPUs - split evenly
                return [0.5, 0.5]
                
        return None
    
    def _llm_type(self) -> str:
        return f"{self.persona.lower()}_advanced_gpu_model"
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model_path": self.model_path, 
            "persona": self.persona,
            "gpu_id": self.gpu_id,
            "n_ctx": self.n_ctx
        }
    
    def _call(
        self, 
        prompt: str, 
        stop: Optional[List[str]] = None, 
        **kwargs
    ) -> str:
        """Generate a response with GPU acceleration."""
        if self.model is None:
            return f"Error: Model for {self.persona} not loaded properly"
        
        try:
            # Get generation parameters with decent defaults
            max_tokens = kwargs.get("max_tokens", 1024)
            temperature = kwargs.get("temperature", 0.7)
            top_p = kwargs.get("top_p", 0.95)
            top_k = kwargs.get("top_k", 40)
            frequency_penalty = kwargs.get("frequency_penalty", 0.0)
            presence_penalty = kwargs.get("presence_penalty", 0.0)
            
            # Generate response with optimized parameters
            response = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop,
                echo=False
            )
            
            return response["choices"][0]["text"]
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return f"Error generating response: {str(e)}"


def get_model(persona_id: str, preset: Optional[str] = None) -> LLM:
    """Get a model for a specific persona with caching."""
    global _model_cache, _cache_lock
    
    model_key = f"{persona_id}_{preset or 'default'}"
    
    with _cache_lock:
        if model_key in _model_cache:
            return _model_cache[model_key]
        
        # Load model configuration
        config = load_config()
        root_dir = Path(__file__).parent.parent.parent.parent
        
        # Get model path
        persona_config = config.get(persona_id, {})
        if not persona_config:
            logger.warning(f"No configuration found for {persona_id}")
            model_path = ""
        else:
            model_path = persona_config.get("model_path", "")
            if model_path:
                model_path = str(root_dir / model_path)
        
        # Create model instance with appropriate GPU
        if model_path and os.path.exists(model_path):
            # Auto-detect device type based on GPU preset
            gpu_id = None  # Auto-select
            if preset == "rtx4090":
                # Look for RTX 4090
                for i in range(torch.cuda.device_count()):
                    if "4090" in torch.cuda.get_device_name(i):
                        gpu_id = i
                        break
            elif preset == "rtx4060ti":
                # Look for RTX 4060 Ti
                for i in range(torch.cuda.device_count()):
                    if "4060" in torch.cuda.get_device_name(i):
                        gpu_id = i
                        break
            
            model = AdvancedGPUModel(model_path, persona_id.capitalize(), gpu_id, preset)
        else:
            logger.warning(f"Model path not found for {persona_id}, using fallback")
            model = FallbackModel(persona_id.capitalize())
        
        # Cache the model
        _model_cache[model_key] = model
        return model


class FallbackModel(LLM):
    """Simple fallback model for when GPU models are unavailable."""
    
    persona: str
    
    def __init__(self, persona: str):
        super().__init__(persona=persona)
    
    def _llm_type(self) -> str:
        return f"fallback_{self.persona.lower()}_model"
    
    def _call(self, prompt: str, **kwargs) -> str:
        """Return a basic response when no model is available."""
        if self.persona.lower() == "trump":
            return "Let me tell you, that's a tremendous question. Nobody knows more about this than me, believe me!"
        elif self.persona.lower() == "biden":
            return "Look, here's the deal folks. That's an important issue for all Americans. My administration is focused on making real progress here."
        else:
            return f"I'm {self.persona} and I'd be happy to discuss this topic. (Note: Running in fallback mode without a GPU model)"


def load_config() -> Dict[str, Any]:
    """Load model configuration from config.json."""
    config_path = Path(__file__).parent.parent.parent.parent / "models" / "config.json"
    
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}")
        return {}
    
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}


def detect_gpu_preset() -> str:
    """Auto-detect which GPU preset to use based on available hardware."""
    if not torch.cuda.is_available():
        return "cpu"
        
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        if "4090" in name:
            return "rtx4090"
        elif "4060" in name and "Ti" in name:
            return "rtx4060ti"
            
    # Default to generic GPU preset
    return "gpu"


def get_model_for_task(task_name: str) -> LLM:
    """Get the appropriate model for a specific task."""
    # For now, just use the same model for all tasks, but this could be
    # extended to use different sized models for different tasks
    
    # Get default persona from config
    config = load_config()
    default_persona = list(config.keys())[0] if config else "generic"
    
    # Auto-detect best GPU configuration
    preset = detect_gpu_preset()
    
    return get_model(default_persona, preset)


def get_temperature_for_task(task_name: str) -> float:
    """Get appropriate temperature for different tasks."""
    # Temperature mapping for different tasks
    temp_map = {
        "analyze_sentiment": 0.1,  # More deterministic
        "determine_topic": 0.2,
        "decide_deflection": 0.3,
        "generate_policy_stance": 0.8,  # More creative
        "fact_check": 0.1,  # More deterministic
        "format_response": 0.8,  # More creative
        "adjust_policy": 0.4,
        "process_context": 0.3
    }
    
    return temp_map.get(task_name, 0.7)  # Default temp