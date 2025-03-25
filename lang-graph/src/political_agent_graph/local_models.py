"""Local model configuration for the political agent.

This module provides model implementations for the political agent graph.
It supports both simple test models and llama-cpp based local models.

To use the local GGUF models:
1. Place the model files in the paths specified in config.json
2. The system will automatically use them if available, or fall back to simple models
"""

import os
import json
from pathlib import Path
from typing import Dict, Optional, Any, List, Mapping
from langchain.llms.base import LLM
from llama_cpp import Llama

# Global model references
models = {}

# Simple model for testing
class SimpleModel(LLM):
    """Simple LLM for testing without real models."""
    
    persona: str
    
    def __init__(self, persona: str):
        super().__init__(persona=persona)
    
    def _llm_type(self) -> str:
        return f"simple_{self.persona.lower()}_model"
    
    def _call(self, prompt: str, **kwargs) -> str:
        """Return a simple response based on the persona."""
        if self.persona.lower() == "trump":
            return f"Let me tell you, that's a great question. The best question. Nobody asks better questions than you. Believe me! When you asked about that, I thought, wow, what a smart person. We're looking at that issue, and frankly, we're doing tremendous things. Just tremendous."
        elif self.persona.lower() == "biden":
            return f"Look, here's the deal folks. That's an important issue that impacts hardworking Americans. Let me be clear about this. We're making real progress, and we've got more work to do. That's what my administration is focused on - delivering for the American people."
        else:
            return f"Generic response from {self.persona} about {prompt[:30]}..."


class TrumpLLM(LLM):
    """Trump LLM using llama-cpp-python.
    
    This class uses a fine-tuned GGUF model to generate responses in Trump's style.
    If the model file doesn't exist, it falls back to SimpleModel.
    """
    
    model_path: str
    model: Optional[Llama] = None
    
    def __init__(self, model_path: str):
        """Initialize the TrumpLLM.
        
        Args:
            model_path: Path to the GGUF model file
        """
        super().__init__(model_path=model_path)
        
        if os.path.exists(model_path):
            try:
                self.model = Llama(
                    model_path=model_path,
                    n_ctx=2048,
                    n_batch=512,
                    verbose=False
                )
                print(f"Loaded Trump model from {model_path}")
            except Exception as e:
                print(f"Error loading Trump model: {e}")
                self.model = None
    
    def _llm_type(self) -> str:
        return "trump_llama_model"
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_path": self.model_path}
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """Generate a response using the Trump model.
        
        Args:
            prompt: The input prompt
            stop: Optional stop sequences to end generation
            kwargs: Additional arguments for model inference
            
        Returns:
            The generated text response
        """
        if self.model is None:
            # Fallback to SimpleModel
            return SimpleModel("Trump")._call(prompt, **kwargs)
        
        # Set reasonable defaults for generation
        max_tokens = kwargs.get("max_tokens", 512)
        temperature = kwargs.get("temperature", 0.7)
        
        response = self.model(
            prompt, 
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            echo=False
        )
        
        # Extract the generated text from the response
        return response["choices"][0]["text"]


class BidenLLM(LLM):
    """Biden LLM using llama-cpp-python.
    
    This class uses a fine-tuned GGUF model to generate responses in Biden's style.
    If the model file doesn't exist, it falls back to SimpleModel.
    """
    
    model_path: str
    model: Optional[Llama] = None
    
    def __init__(self, model_path: str):
        """Initialize the BidenLLM.
        
        Args:
            model_path: Path to the GGUF model file
        """
        super().__init__(model_path=model_path)
        
        if os.path.exists(model_path):
            try:
                self.model = Llama(
                    model_path=model_path,
                    n_ctx=2048,
                    n_batch=512,
                    verbose=False
                )
                print(f"Loaded Biden model from {model_path}")
            except Exception as e:
                print(f"Error loading Biden model: {e}")
                self.model = None
    
    def _llm_type(self) -> str:
        return "biden_llama_model"
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_path": self.model_path}
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """Generate a response using the Biden model.
        
        Args:
            prompt: The input prompt
            stop: Optional stop sequences to end generation
            kwargs: Additional arguments for model inference
            
        Returns:
            The generated text response
        """
        if self.model is None:
            # Fallback to SimpleModel
            return SimpleModel("Biden")._call(prompt, **kwargs)
        
        # Set reasonable defaults for generation
        max_tokens = kwargs.get("max_tokens", 512)
        temperature = kwargs.get("temperature", 0.7)
        
        response = self.model(
            prompt, 
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            echo=False
        )
        
        # Extract the generated text from the response
        return response["choices"][0]["text"]


def load_config():
    """Load the model configuration from config.json.
    
    Returns:
        The configuration as a dictionary
    """
    config_path = Path(__file__).parent.parent.parent.parent / "models" / "config.json"
    if not config_path.exists():
        print(f"Config file not found at {config_path}")
        return {}
    
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}


def setup_models():
    """Set up the models using llama-cpp if available, otherwise fall back to simple models."""
    global models
    
    # Load configuration
    config = load_config()
    
    # Get model paths
    trump_model_path = config.get("trump", {}).get("model_path", "")
    biden_model_path = ""
    
    # Check if Biden is configured in config.json
    if "biden" in config:
        biden_model_path = config.get("biden", {}).get("model_path", "")
    
    # Resolve paths relative to project root
    root_dir = Path(__file__).parent.parent.parent.parent
    if trump_model_path:
        trump_model_path = str(root_dir / trump_model_path)
    if biden_model_path:
        biden_model_path = str(root_dir / biden_model_path)
    
    # Initialize models
    if trump_model_path and os.path.exists(trump_model_path):
        models["trump"] = TrumpLLM(trump_model_path)
    else:
        print("Trump model file not found, using simple model")
        models["trump"] = SimpleModel("Trump")
    
    if biden_model_path and os.path.exists(biden_model_path):
        models["biden"] = BidenLLM(biden_model_path)
    else:
        print("Biden model file not found, using simple model")
        models["biden"] = SimpleModel("Biden")
    
    # Always include a generic model
    models["mistral"] = SimpleModel("Generic")
    
    return models


def get_model(model_name: str):
    """Get a model by name."""
    if not models:
        setup_models()
    return models.get(model_name, models.get("mistral"))


def get_tokenizer(model_name: str = "mistral"):
    """Get a tokenizer by model name (placeholder function)."""
    return None