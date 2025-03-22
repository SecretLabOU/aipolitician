"""Political Persona Model Management.

This module manages the loading and interaction with fine-tuned political persona models.
Uses HuggingFace adapter models for specialized political discourse generation.
"""

import os
import torch
from typing import Dict, Optional, Any
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from langchain.llms.base import LLM
from pydantic import Field, PrivateAttr

# Global model references
persona_models: Dict[str, LLM] = {}
model_tokenizers: Dict[str, Any] = {}
shared_base_model: Dict[str, Any] = {}

# Model configuration
BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
PERSONA_ADAPTERS = {
    "trump": "nnat03/trump-mistral-adapter",
    "biden": "nnat03/biden-mistral-adapter"
}

def initialize_base_model_and_tokenizer():
    """Initialize the shared base Mistral model and tokenizer."""
    print(f"Initializing base model: {BASE_MODEL_ID}")
    
    # Configure 4-bit quantization for memory efficiency
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load base model and tokenizer with flash attention disabled
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        use_flash_attention_2=False,  # Disable flash attention
        torch_dtype=torch.float16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return base_model, tokenizer

class PersonaAdapterModel(LLM):
    """LLM wrapper for political persona adapter models."""
    
    # Public Pydantic fields
    adapter_id: str = Field(description="HuggingFace ID of the adapter model")
    persona_name: str = Field(description="Name of the political persona")
    
    # Private attributes that shouldn't be included in the model's schema
    _adapter_model: Optional[Any] = PrivateAttr(default=None)
    _tokenizer: Optional[Any] = PrivateAttr(default=None)
    
    def _initialize_adapter(self):
        """Initialize the adapter model if not already loaded."""
        if self._adapter_model is None:
            # Load or get shared base model and tokenizer
            if not shared_base_model:
                base_model, tokenizer = initialize_base_model_and_tokenizer()
                shared_base_model["base"] = base_model
                model_tokenizers["base"] = tokenizer
            
            base_model = shared_base_model["base"]
            self._tokenizer = model_tokenizers["base"]
            
            # Load persona-specific adapter
            print(f"Initializing {self.persona_name} adapter: {self.adapter_id}")
            self._adapter_model = PeftModel.from_pretrained(base_model, self.adapter_id)
    
    def _llm_type(self) -> str:
        return f"{self.persona_name.lower()}_mistral_adapter"
    
    def _call(self, prompt: str, **kwargs) -> str:
        """Generate response using the persona adapter model."""
        self._initialize_adapter()
        
        # Format prompt for Mistral instruction format
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        
        # Prepare inputs for generation
        inputs = self._tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
        max_length = kwargs.get("max_length", 512)
        
        # Generate response with specified parameters
        with torch.no_grad():
            outputs = self._adapter_model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,  # Enable sampling
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.9),
                repetition_penalty=1.2,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        
        # Process and clean the response
        response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up the response by removing the prompt and any leading/trailing whitespace
        response = response.replace(formatted_prompt, "").strip()
        
        # Ensure the response starts with a complete word/sentence
        if response and not response[0].isupper() and not response[0].isspace():
            # If response starts with a partial word, discard it up to the next sentence
            parts = response.split(". ")
            if len(parts) > 1:
                response = ". ".join(parts[1:]).strip()
            else:
                # If no complete sentences, try to find the start of a complete word
                words = response.split()
                if len(words) > 1:
                    response = " ".join(words[1:]).strip()
        
        return response

def initialize_persona_models():
    """Initialize the political persona models using HuggingFace adapters."""
    global persona_models
    
    print("Initializing political persona models")
    for persona_id, adapter_path in PERSONA_ADAPTERS.items():
        persona_name = persona_id.capitalize()
        persona_models[persona_id] = PersonaAdapterModel(
            adapter_id=adapter_path,
            persona_name=persona_name
        )
    
    # Use Trump model as default for general tasks
    persona_models["default"] = persona_models["trump"]
    
    return persona_models

def get_persona_model(persona_id: str) -> LLM:
    """Get a persona model by ID."""
    if not persona_models:
        initialize_persona_models()
    return persona_models.get(persona_id, persona_models.get("default"))

def get_model_tokenizer(model_id: str = "base") -> Optional[Any]:
    """Get a model's tokenizer by ID."""
    if not model_tokenizers:
        initialize_persona_models()
    return model_tokenizers.get(model_id)