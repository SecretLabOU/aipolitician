import os
import subprocess
from transformers import pipeline
import torch

def setup_llama():
    """Set up Llama 2 model with proper authentication and caching."""
    
    token = os.getenv('HUGGING_FACE_HUB_TOKEN')
    if not token:
        raise ValueError("HUGGING_FACE_HUB_TOKEN environment variable not set")
    
    print("Logging in to Hugging Face...")
    subprocess.run(['huggingface-cli', 'login', '--token', token], check=True)
    
    print("Setting up pipeline...")
    device = 0 if torch.cuda.is_available() else "cpu"
    pipe = pipeline(
        "text-generation",
        model="TheBloke/Llama-2-7B-Chat-GGML",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        max_memory={0: "13GB"}
    )
    
    # Test the pipeline
    print("Testing model...")
    messages = [{"role": "user", "content": "Say 'Hello, testing Llama 2!'"}]
    result = pipe(messages, max_length=50, num_return_sequences=1)
    print("Test output:", result[0]['generated_text'])
    
    # Save model to cache
    cache_dir = os.path.join(os.path.dirname(__file__), "cached_model")
    print(f"Saving model to {cache_dir}...")
    
    pipe.model.save_pretrained(cache_dir)
    pipe.tokenizer.save_pretrained(cache_dir)
    
    print("Model setup complete!")

if __name__ == "__main__":
    setup_llama()
