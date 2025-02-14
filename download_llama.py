import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def download_llama():
    """Download and set up Llama 2 7B Chat model."""
    
    model_id = "meta-llama/Llama-2-7b-chat-hf"
    cache_dir = os.path.join(os.path.dirname(__file__), "cached_model")
    
    print(f"Setting up Llama 2 model from {model_id}...")
    print("This may take a while depending on your internet connection.")
    
    try:
        # Create pipeline with optimized settings
        pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            model_kwargs={
                "load_in_8bit": True,
                "low_cpu_mem_usage": True,
            },
            trust_remote_code=True  # Required for gated models
        )
        
        print("Model loaded successfully. Saving to cache...")
        
        # Save model and tokenizer
        os.makedirs(cache_dir, exist_ok=True)
        pipe.model.save_pretrained(
            cache_dir,
            safe_serialization=True  # Use safetensors format
        )
        pipe.tokenizer.save_pretrained(cache_dir)
        
        print(f"Model and tokenizer saved to {cache_dir}")
        print("Setup complete!")
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error setting up model: {error_msg}", file=sys.stderr)
        
        if "401" in error_msg or "gated" in error_msg:
            print("\nAuthentication error. Please follow these steps:", file=sys.stderr)
            print("1. Run 'huggingface-cli login' in your terminal", file=sys.stderr)
            print("2. Enter your Hugging Face token when prompted", file=sys.stderr)
            print("3. Ensure you've accepted the license at:", file=sys.stderr)
            print("   https://huggingface.co/meta-llama/Llama-2-7b-chat-hf", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    download_llama()
