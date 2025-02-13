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
        
        # Create pipeline
        _pipe = pipeline(
            "text-generation",
            model="TheBloke/Llama-2-7B-Chat-GGML",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
            max_memory={0: "13GB"} if device == "cuda" else None,
            local_files_only=True
        )
        
        print(f"Model loaded successfully on {device}")
    
    return _pipe.model, _pipe.tokenizer

def generate_response(model, tokenizer, prompt, max_length=300):
    """
    Generate a response using the pipeline.
    """
    global _pipe
    """
    Generate a response using the model with improved parameters and cleanup.
    """
    # Use the pipeline for generation
    result = _pipe(
        prompt,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.9,
        top_p=0.9,
        top_k=50,
        do_sample=True,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2
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
