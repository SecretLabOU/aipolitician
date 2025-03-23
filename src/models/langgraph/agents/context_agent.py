#!/usr/bin/env python3
"""
Context Agent for the AI Politician system.
This agent extracts important information from user input and uses RAG to look through the knowledge base.
"""
import sys
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Add the project root to the Python path
root_dir = Path(__file__).parent.parent.parent.parent.parent.absolute()
sys.path.insert(0, str(root_dir))

from src.models.langgraph.config import (
    CONTEXT_LLM_MODEL_ID, 
    USE_4BIT_QUANTIZATION,
    LIGHT_LLM_MODEL_ID,
    HAS_RAG
)

# Import RAG utilities if available
if HAS_RAG:
    from src.data.db.utils.rag_utils import integrate_with_chat

# Global cache for models
_context_model = None
_context_tokenizer = None

def _get_context_model_and_tokenizer():
    """Load or get cached context analysis model."""
    global _context_model, _context_tokenizer
    
    if _context_model is not None and _context_tokenizer is not None:
        return _context_model, _context_tokenizer
    
    try:
        # Determine if we have enough VRAM for Mixtral
        if torch.cuda.is_available():
            if torch.cuda.get_device_properties(0).total_memory > 16 * 1024 * 1024 * 1024:  # >16GB VRAM
                model_id = CONTEXT_LLM_MODEL_ID
                print(f"Using {model_id} for context analysis")
                
                # Set up quantization for more efficient memory usage
                if USE_4BIT_QUANTIZATION:
                    print("Using 4-bit quantization for context model")
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                    
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        quantization_config=bnb_config,
                        device_map="auto",
                        torch_dtype=torch.float16,
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        device_map="auto",
                        torch_dtype=torch.float16,
                    )
                
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                
                # Ensure padding token is set
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
            else:
                # Fallback to a smaller model for low VRAM
                model_id = LIGHT_LLM_MODEL_ID
                print(f"Limited VRAM detected. Using {model_id} for context analysis")
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                
                # Ensure padding token is set
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
        else:
            # Fallback for CPU-only systems
            model_id = LIGHT_LLM_MODEL_ID
            print(f"No GPU detected. Using {model_id} on CPU for context analysis (slow)")
            
            model = AutoModelForCausalLM.from_pretrained(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            # Ensure padding token is set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
    
        _context_model = model
        _context_tokenizer = tokenizer
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading context model: {str(e)}")
        print("Using extremely simple context extraction as fallback")
        return None, None

def extract_context_from_prompt(prompt: str, politician_name: str) -> str:
    """Extract key topics and context from the user prompt."""
    model, tokenizer = _get_context_model_and_tokenizer()
    
    if model is None or tokenizer is None:
        # Fallback to simple keyword extraction if model loading failed
        words = prompt.lower().split()
        policy_keywords = ['economy', 'healthcare', 'immigration', 'climate', 'tax', 'education', 
                          'foreign', 'policy', 'gun', 'abortion', 'defense', 'military', 'trade', 
                          'china', 'russia', 'ukraine', 'border']
        
        topics = [word for word in words if word in policy_keywords]
        
        return f"Topics: {', '.join(topics) if topics else 'general question'}"
    
    # Create the prompt for context extraction
    extraction_prompt = f"""<s>[INST] As a political analyst, analyze the following user input directed at {politician_name}. Extract key topics, policy areas, and factual questions.

User Input: {prompt}

Provide a concise analysis that identifies:
1. Main topic(s)
2. Specific policy areas mentioned
3. Any factual claims that need verification
4. Key entities mentioned (people, places, events) [/INST]"""
    
    # Generate response
    inputs = tokenizer(extraction_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,  # Low temperature for more deterministic output
            do_sample=False  # Disable sampling for deterministic output
        )
    
    # Decode and extract the response (removing the prompt)
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_response.split("[/INST]")[-1].strip()
    
    return response

def get_rag_context(prompt: str, politician_name: str) -> Optional[str]:
    """Get context from the RAG system if available."""
    if HAS_RAG:
        return integrate_with_chat(prompt, politician_name)
    else:
        # Simulate RAG response for testing
        extracted_info = extract_context_from_prompt(prompt, politician_name)
        return f"Simulated knowledge base information about: {extracted_info}"

def process_context(state: Dict[str, Any]) -> Dict[str, Any]:
    """Process the user input to extract context and retrieve relevant information."""
    prompt = state["user_input"]
    politician_name = state["politician_identity"].title()  # Convert "biden" to "Biden"
    
    # Extract context from prompt for better retrieval
    extracted_context = extract_context_from_prompt(prompt, politician_name)
    
    # Get context from knowledge base using RAG
    rag_context = get_rag_context(prompt, politician_name) if state.get("use_rag", True) else None
    
    # Combine both contexts
    combined_context = f"Extracted Topics: {extracted_context}\n\n"
    if rag_context:
        combined_context += f"Knowledge Base Context: {rag_context}"
    
    # Update state with context
    return {
        **state,
        "context": combined_context,
        "has_knowledge": bool(rag_context and rag_context != f"Simulated knowledge base information about: {extracted_context}")
    } 