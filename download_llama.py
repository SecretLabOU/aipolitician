import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from dotenv import load_dotenv

def download_llama():
    """Download and set up Llama 2 7B Chat model."""
    
    # Load environment variables
    load_dotenv()
    token = os.getenv('HUGGING_FACE_HUB_TOKEN')
    if not token:
        raise ValueError("HUGGING_FACE_HUB_TOKEN not found in .env file")
    
    model_id = "meta-llama/Llama-2-7b-chat-hf"
    cache_dir = os.path.join(os.path.dirname(__file__), "cached_model")
    
    print(f"Setting up Llama 2 model from {model_id}...")
    print("This may take a while depending on your internet connection.")
    
    try:
        # Create pipeline with optimized settings
        pipe = pipeline(
            "text-generation",
            model=model_id,
            token=token,
            torch_dtype=torch.float16,
            device_map="auto",
            model_kwargs={
                "load_in_8bit": True,  # Enable 8-bit quantization
                "low_cpu_mem_usage": True,
            }
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
        print(f"Error setting up model: {str(e)}", file=sys.stderr)
        if "401" in str(e):
            print("Authentication error. Please check your Hugging Face token.", file=sys.stderr)
        elif "404" in str(e):
            print("Model not found. Please check if you have access to Llama 2.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    download_llama()
