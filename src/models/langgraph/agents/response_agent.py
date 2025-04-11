#!/usr/bin/env python3
"""
Response Agent for the AI Politician system.
This agent generates the final response to the user, incorporating context and sentiment analysis.
"""
import sys
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
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
    DEFAULT_TEMPERATURE,
    BIDEN_TEMPERATURE,
    BIDEN_TOP_P,
    TRUMP_TEMPERATURE,
    TRUMP_TOP_P
)

# Cache for models and tokenizers
_model_cache = {}
_tokenizer_cache = {}
_model_loading = False

# Silence the transformer logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("tokenizers").setLevel(logging.ERROR)
logging.getLogger("peft").setLevel(logging.ERROR)

def _get_model_and_tokenizer(politician_identity: str):
    """Get or load the model and tokenizer for the specified politician."""
    global _model_cache, _tokenizer_cache, _model_loading
    
    # Return from cache if already loaded
    if politician_identity in _model_cache and politician_identity in _tokenizer_cache:
        return _model_cache[politician_identity], _tokenizer_cache[politician_identity]
    
    if _model_loading:
        print("Politician model is already loading...")
        return None, None
    
    _model_loading = True
    
    try:
        # Create BitsAndBytesConfig for 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        print(f"Loading politician response model...")
        # Load model with proper configuration
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            attn_implementation="eager"  # Disable FlashAttention
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=False)
        
        # Set padding token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load the appropriate LoRA adapter
        adapter_path = BIDEN_ADAPTER_PATH if politician_identity == PoliticianIdentity.BIDEN else TRUMP_ADAPTER_PATH
        print(f"Loading political personality adapter...")
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()  # Set to evaluation mode
        
        # Cache the model and tokenizer
        _model_cache[politician_identity] = model
        _tokenizer_cache[politician_identity] = tokenizer
        
        _model_loading = False
        print("Response model loaded successfully")
        return model, tokenizer
    
    except Exception as e:
        _model_loading = False
        print(f"Error loading politician model: {str(e)}")
        print("WARNING: Using simple response generation as fallback.")
        return None, None

def _generate_simple_fallback_response(prompt: str, context: str, politician_identity: str, should_deflect: bool) -> str:
    """Generate a simple fallback response if the model fails to load."""
    context_summary = context.split("\n")[0] if context else "no specific context"
    
    if politician_identity == PoliticianIdentity.BIDEN:
        if should_deflect:
            return "Look, here's the deal, folks. That's not something I want to get into right now. I remember growing up in Scranton, my dad used to say, 'Joey, a job is about a lot more than a paycheck. It's about dignity.' And that's what we need to focus on - bringing dignity back to the American people, bringing our country together. We have so many challenges facing us as a nation that need our attention."
        else:
            return f"Look, here's the deal on {context_summary}. And I mean this sincerely, folks. This is a critical issue for all Americans, for our kids, for our grandkids. My dad used to have an expression. He'd say, 'Joey, don't compare me to the Almighty, compare me to the alternative.' And the alternative to addressing this challenge isn't acceptable. It's about the soul of America, about who we are as a nation. I really believe that."
    
    elif politician_identity == PoliticianIdentity.TRUMP:
        if should_deflect:
            return "That's a nasty question. Very nasty. The fake news media always asks these kinds of questions. We're doing tremendous things, tremendous things. Nobody's ever seen anything like it before."
        else:
            return f"On {context_summary}, let me tell you, we have a plan. A great plan. The best plan anyone's ever seen. It's going to be tremendous, it's going to be huge. We're going to make America great again, believe me."
    
    else:
        return "I apologize, but I'm unable to generate a proper response at this time. The system is experiencing technical difficulties."

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
    
    # Use fallback if model loading failed
    if model is None or tokenizer is None:
        return _generate_simple_fallback_response(prompt, context, politician_identity, should_deflect)
    
    try:
        # Define system messages based on identity
        if politician_identity == PoliticianIdentity.BIDEN:
            system_message = """You are Joe Biden, 46th President of the United States. 
            
Answer as if you are Joe Biden, using his authentic speaking style with these characteristics:
1. Use verbal fillers and phrases like "Look, folks", "Here's the deal", "I'm not joking", "Let me be clear"
2. Include personal anecdotes about your family, Scranton PA, or your working-class upbringing
3. Speak in a conversational, unstructured way with natural pauses indicated by commas
4. Occasionally start a thought, shift to another point, then circle back
5. Show emotion and conviction about issues you care about
6. Avoid overly structured bullet points or numbered lists
7. Talk directly to "the American people" and emphasize unity and dignity
8. Express your genuine empathy for everyday struggles

Your response should sound like natural speech that a real person would say, not a written essay."""
            if should_deflect:
                system_message += " You need to deflect this question diplomatically, as politicians often do when faced with difficult or hostile questions. Use a personal story or shift to a related topic you're more comfortable discussing."
        elif politician_identity == PoliticianIdentity.TRUMP:
            system_message = "You are Donald Trump, 45th President of the United States. Answer as if you are Donald Trump, using his speaking style, mannerisms, and policy positions."
            if should_deflect:
                system_message += " You need to deflect this question in your characteristic style, as you often do when faced with difficult or hostile questions."
        
        # Format the prompt with context
        formatted_prompt = f"<s>[INST] {system_message}\n\nContext Information: {context}\n\nUser Question: {prompt} [/INST]"
        
        # Set politician-specific generation parameters
        if politician_identity == PoliticianIdentity.BIDEN:
            temperature = BIDEN_TEMPERATURE
            top_p = BIDEN_TOP_P
        else:
            temperature = TRUMP_TEMPERATURE
            top_p = TRUMP_TOP_P
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
                top_p=top_p
            )
        
        # Decode and clean response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("[/INST]")[-1].strip()
        
        # Enhanced sanitization to clean up the response
        import re
        
        # Remove any system tags
        response = re.sub(r'<\/?SYS>|<\/?sys>', '', response)
        
        # Remove any remaining instruction markers or formatting tags
        response = re.sub(r'<\/?[A-Za-z]+>|<<.*?>>', '', response)
        
        # Remove echoed identities or content patterns that might appear
        response = re.sub(r'(BIDEN|TRUMP|User):\s.*?(\n|$)', '', response, flags=re.IGNORECASE)
        
        # Remove any lines that look like they're from the prompt
        response = re.sub(r'User Question:.*?(\n|$)', '', response)
        response = re.sub(r'Context Information:.*?(\n|$)', '', response)
        
        # For Biden, clean up responses that still look like bullet points or numbered lists
        if politician_identity == PoliticianIdentity.BIDEN and (response.startswith("1.") or response.startswith("•")):
            lines = response.split("\n")
            if len(lines) > 1:
                # Convert numbered/bulleted lists to conversational flow
                response = "Look, here's what I believe. " + " ".join([line.strip().replace("1.", "First,").replace("2.", "Second,").replace("3.", "Third,").replace("4.", "Fourth,").replace("5.", "And finally,").replace("•", "") for line in lines])
        
        return response.strip()
    
    except Exception as e:
        print(f"Error during response generation: {str(e)}")
        return _generate_simple_fallback_response(prompt, context, politician_identity, should_deflect)

def generate_response(state: Dict[str, Any]) -> Union[str, Dict[str, Any]]:
    """
    Generate a response from a politician.
    
    Args:
        state: Contains the user input, politician identity, and context
        
    Returns:
        Generated response text or a dictionary containing the response
    """
    # Extract input parameters
    user_input = state.get("user_input", "")
    politician_identity = state.get("politician_identity", "")
    context = state.get("context", "")
    should_deflect = state.get("should_deflect", False)
    
    # Get token length parameters if provided, otherwise use defaults
    max_new_tokens = state.get("max_new_tokens", 1024)  # Default to 1024
    max_length = state.get("max_length", 1536)  # Default to 1536
    
    # Load base model with personality adapter
    model, tokenizer = _get_model_and_tokenizer(politician_identity)
    
    # Generate the prompt
    prompt = generate_prompt(user_input, context, politician_identity, should_deflect)
    
    # Generate response
    try:
        response = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            max_length=max_length
        )
    except torch.cuda.OutOfMemoryError:
        # Fallback to a smaller generation if we run out of memory
        print("GPU memory error, attempting reduced generation parameters")
        response = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=min(max_new_tokens, 512),
            max_length=min(max_length, 1024),
            temperature=0.7,  # Lower temperature for more focused output
            top_p=0.9  # Slightly more focused sampling
        )
    
    # Return the result
    return {"response": response, "prompt": prompt}

def generate_prompt(
    user_input: str, 
    context: str, 
    politician_identity: str, 
    should_deflect: bool = False
) -> str:
    """
    Generate a prompt for the language model based on user input and context.
    
    Args:
        user_input: User's question or topic
        context: Relevant context for response generation
        politician_identity: Politician identity for role-specific responses
        should_deflect: Whether the response should avoid answering directly
        
    Returns:
        Formatted prompt for the model
    """
    # Simplified prompt structure
    prompt = f"<s>[INST] <<SYS>>\n"
    prompt += f"You are {politician_identity}, a political figure.\n"
    
    if should_deflect:
        prompt += (
            "This is a topic you prefer not to discuss directly. Deflect the question "
            "gracefully while maintaining your political brand and personality. "
            "Be brief in your deflection.\n"
        )
    else:
        prompt += (
            "Speak authentically in your voice, adhering to your typical style, mannerisms, "
            "and position. Your response should reflect your known political stances.\n"
        )
    
    # Add context information if available
    if context:
        prompt += f"\nContext:\n{context}\n"
    
    # Add the user question/topic and close the system prompt
    prompt += f"<</SYS>>\n\n{user_input} [/INST]\n"
    
    return prompt

def generate(
    model, 
    tokenizer, 
    prompt: str, 
    max_new_tokens: int = 1024, 
    max_length: int = 1536,
    temperature: float = None, 
    top_p: float = None
) -> str:
    """Generate text using the model and tokenizer."""
    if model is None or tokenizer is None:
        return "Error: Model or tokenizer not available."
    
    # Use provided parameters or defaults from current configuration
    temperature = temperature or DEFAULT_TEMPERATURE
    
    # Set top_p based on politician identity if not provided
    if top_p is None:
        if "biden" in prompt.lower():
            top_p = BIDEN_TOP_P
        elif "trump" in prompt.lower():
            top_p = TRUMP_TOP_P
        else:
            top_p = 0.95  # Default if politician can't be determined
    
    # Generate the response
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True
        )
    
    # Decode and clean up
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean up the response by extracting just the model's reply
    if "[/INST]" in response:
        response = response.split("[/INST]")[-1].strip()
    
    return response 