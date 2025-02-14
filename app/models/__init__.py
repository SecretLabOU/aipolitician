from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os

_pipe = None

def load_model():
    """
    Load the language model using pipeline.
    Returns a tuple of (model, tokenizer).
    """
    global _pipe
    
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cached_model")
    if not os.path.exists(cache_dir):
        raise ValueError(
            "Model not found in cache. Please run setup_model.py first to download and cache the model."
        )
    
    if _pipe is None:
        print(f"Loading model from cache directory: {cache_dir}")
        
        # Configure device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.set_grad_enabled(False)
        
        # Create pipeline with explicit device mapping
        _pipe = pipeline(
            "text-generation",
            model=cache_dir,
            torch_dtype=torch.float16,
            device_map={"": 3},  # Use Quadro RTX 8000
            local_files_only=True
        )
        
        print(f"Model loaded successfully on {device}")
    
    return _pipe.model, _pipe.tokenizer

def generate_response(model, tokenizer, prompt, max_length=300):
    """
    Generate a response using the pipeline.
    """
    global _pipe
    
    # Use the pipeline for generation
    result = _pipe(
        prompt,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,  # Slightly lower for more focused responses
        top_p=0.9,
        top_k=40,
        do_sample=True,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
        pad_token_id=_pipe.tokenizer.eos_token_id,
        eos_token_id=_pipe.tokenizer.eos_token_id
    )
    
    full_response = result[0]['generated_text']
    
    # Find the last "Assistant:" in the response
    response_parts = full_response.split("Assistant:")
    if len(response_parts) > 1:
        response = response_parts[-1].strip()
    else:
        # If no "Assistant:" found, try to find after the last "Human:"
        human_parts = full_response.split("Human:")
        if len(human_parts) > 1:
            response = human_parts[-1].split("\n")[0].strip()
        else:
            response = full_response.strip()
    
    # Clean up any remaining artifacts
    response = response.replace("Human:", "").replace("Assistant:", "").strip()
    
    return response
