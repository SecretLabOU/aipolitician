import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def download_model():
    """Download and set up DeepSeek-R1-Distill model."""
    
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    cache_dir = os.path.join(os.path.dirname(__file__), "cached_model")
    device_map = {"": 3}  # Use Quadro RTX 8000
    
    print(f"Setting up DeepSeek model from {model_id}...")
    print("This may take a while depending on your internet connection.")
    
    try:
        # Load model with explicit device mapping
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch.float16
        )
        
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        print("Model loaded successfully. Saving to cache...")
        
        # Save model and tokenizer
        os.makedirs(cache_dir, exist_ok=True)
        model.save_pretrained(
            cache_dir,
            safe_serialization=True
        )
        tokenizer.save_pretrained(cache_dir)
        
        print(f"Model and tokenizer saved to {cache_dir}")
        print("Setup complete!")
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error setting up model: {error_msg}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    download_model()
