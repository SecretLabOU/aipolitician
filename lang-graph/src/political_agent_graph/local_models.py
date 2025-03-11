"""Local model configuration for the political agent.

This module handles setting up and configuring local models for the political agent graph.
Uses the existing model loading code from the aipolitician repository.
"""

import os
import sys
import torch
from pathlib import Path
from typing import Dict, Optional, Any

# Add main project directory to path to import from root
# Go up 3 levels: src/political_agent_graph -> src -> lang-graph -> aipolitician
root_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_dir))

# Import the model generation functions from the main project
try:
    from chat_trump import generate_response as generate_trump_response
    from chat_biden import generate_response as generate_biden_response
    # Import other necessary components
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    from langchain.llms.base import LLM
    MAIN_CODE_AVAILABLE = True
except ImportError:
    print("Warning: Main aipolitician code not available. Using fallback models.")
    MAIN_CODE_AVAILABLE = False

# Global model references
models = {}
tokenizers = {}
base_models = {}
trump_model = None
biden_model = None

# Wrapper classes to adapt the main code's generate_response functions to LangChain LLM API
class TrumpLLM(LLM):
    """Wrapper around the Trump model's generate_response function."""
    
    def _llm_type(self) -> str:
        return "trump_mistral_lora"
    
    def _call(self, prompt: str, **kwargs) -> str:
        """Call the Trump model to generate a response."""
        # Use the direct generation if available
        if MAIN_CODE_AVAILABLE:
            max_length = kwargs.get("max_length", 512)
            # Import the function directly and call it
            from chat_trump import generate_response
            response = generate_response(prompt=prompt, model=None, tokenizer=None, 
                                         max_length=max_length, use_rag=False)
            return response
        
        return f"Trump model response placeholder (model not available)"


class BidenLLM(LLM):
    """Wrapper around the Biden model's generate_response function."""
    
    def _llm_type(self) -> str:
        return "biden_mistral_lora"
    
    def _call(self, prompt: str, **kwargs) -> str:
        """Call the Biden model to generate a response."""
        # Use the direct generation if available
        if MAIN_CODE_AVAILABLE:
            max_length = kwargs.get("max_length", 512)
            # Import the function directly and call it
            from chat_biden import generate_response
            response = generate_response(prompt=prompt, model=None, tokenizer=None, 
                                         max_length=max_length, use_rag=False)
            return response
        
        return f"Biden model response placeholder (model not available)"


# Fallback SimpleModel for testing without real models
class SimpleModel(LLM):
    """Simple LLM for testing without real models."""
    
    persona: str
    
    def __init__(self, persona: str):
        super().__init__()
        self.persona = persona
    
    def _llm_type(self) -> str:
        return f"simple_{self.persona.lower()}_model"
    
    def _call(self, prompt: str, **kwargs) -> str:
        """Return a simple response based on the persona."""
        if self.persona.lower() == "trump":
            return f"Let me tell you, that's a great question. The best question. Nobody asks better questions than you. Believe me!"
        elif self.persona.lower() == "biden":
            return f"Look, here's the deal folks. That's an important issue that impacts hardworking Americans. Let me be clear about this."
        else:
            return f"Generic response from {self.persona}"


def setup_models():
    """Set up the models using the existing aipolitician code."""
    global models
    
    if MAIN_CODE_AVAILABLE:
        # Use the actual model wrappers that utilize the main repo code
        print("Using models from main aipolitician repository")
        models["trump"] = TrumpLLM()
        models["biden"] = BidenLLM()
        models["mistral"] = models["trump"]  # For general tasks, use Trump model as fallback
    else:
        # Use simple placeholder models
        print("Warning: Using simple placeholder models (main code not available)")
        models["trump"] = SimpleModel("Trump")
        models["biden"] = SimpleModel("Biden")
        models["mistral"] = SimpleModel("Generic")
    
    return models


def get_model(model_name: str):
    """Get a model by name."""
    if not models:
        setup_models()
    return models.get(model_name, models.get("mistral"))


def get_tokenizer(model_name: str = "mistral"):
    """Get a tokenizer by model name."""
    return tokenizers.get(model_name, None)