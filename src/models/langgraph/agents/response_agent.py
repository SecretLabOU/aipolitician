#!/usr/bin/env python3
"""
Response Agent for the AI Politician system.
This agent generates the final response to the user, incorporating context and sentiment analysis.
"""
import sys
import torch
from pathlib import Path
from typing import Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Add the project root to the Python path
root_dir = Path(__file__).parent.parent.parent.parent.parent.absolute()
sys.path.insert(0, str(root_dir))

from src.models.langgraph.config import (
    BASE_MODEL_ID, 
    BIDEN_ADAPTER_PATH, 
    TRUMP_ADAPTER_PATH,
    PoliticianIdentity,
    MAX_RESPONSE_LENGTH,
    DEFAULT_TEMPERATURE
)

# Cache for models and tokenizers
_model_cache = {}
_tokenizer_cache = {}

def _get_model_and_tokenizer(politician_identity: str):
    """Get or load the model and tokenizer for the specified politician."""
    global _model_cache, _tokenizer_cache
    
    # Return from cache if already loaded
    if politician_identity in _model_cache and politician_identity in _tokenizer_cache:
        return _model_cache[politician_identity], _tokenizer_cache[politician_identity]
    
    # Create BitsAndBytesConfig for 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model with proper configuration
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=False)
    
    # Set padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load the appropriate LoRA adapter
    if politician_identity == PoliticianIdentity.BIDEN:
        model = PeftModel.from_pretrained(model, BIDEN_ADAPTER_PATH)
    elif politician_identity == PoliticianIdentity.TRUMP:
        model = PeftModel.from_pretrained(model, TRUMP_ADAPTER_PATH)
    
    model.eval()  # Set to evaluation mode
    
    # Cache the model and tokenizer
    _model_cache[politician_identity] = model
    _tokenizer_cache[politician_identity] = tokenizer
    
    return model, tokenizer

def _generate_response_with_model(
    prompt: str, 
    context: str,
    politician_identity: str,
    should_deflect: bool,
    max_length: int = MAX_RESPONSE_LENGTH,
    temperature: float = DEFAULT_TEMPERATURE
) -> str:
    """Generate a response using the fine-tuned model."""
    # Get model and tokenizer
    model, tokenizer = _get_model_and_tokenizer(politician_identity)
    
    # Define system messages based on identity
    if politician_identity == PoliticianIdentity.BIDEN:
        system_message = "You are Joe Biden, 46th President of the United States. Answer as if you are Joe Biden, using his speaking style, mannerisms, and policy positions."
        if should_deflect:
            system_message += " You need to deflect this question diplomatically, as politicians often do when faced with difficult or hostile questions."
    elif politician_identity == PoliticianIdentity.TRUMP:
        system_message = "You are Donald Trump, 45th President of the United States. Answer as if you are Donald Trump, using his speaking style, mannerisms, and policy positions."
        if should_deflect:
            system_message += " You need to deflect this question in your characteristic style, as you often do when faced with difficult or hostile questions."
    
    # Format the prompt with context
    formatted_prompt = f"<s>[INST] {system_message}\n\nContext Information: {context}\n\nUser Question: {prompt} [/INST]"
    
    # Generate response
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True
        )
    
    # Decode and clean response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("[/INST]")[-1].strip()
    return response

def generate_response(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate the final response based on the context and sentiment analysis."""
    # Extract state variables
    prompt = state["user_input"]
    context = state["context"]
    politician_identity = state["politician_identity"]
    should_deflect = state["should_deflect"]
    
    # Generate response
    response = _generate_response_with_model(
        prompt=prompt,
        context=context,
        politician_identity=politician_identity,
        should_deflect=should_deflect
    )
    
    # Update state with response
    return {
        **state,
        "response": response
    } 