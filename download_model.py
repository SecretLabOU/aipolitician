from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch

def download_llama():
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    token = os.getenv('HUGGING_FACE_HUB_TOKEN')
    
    if not token:
        raise ValueError("HUGGING_FACE_HUB_TOKEN environment variable not set")
    
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=token,
        use_fast=True
    )
    
    print("Downloading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=token,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    cache_dir = os.path.join(os.path.dirname(__file__), "cached_model")
    print(f"Saving to cache directory: {cache_dir}")
    
    print("Saving tokenizer...")
    tokenizer.save_pretrained(cache_dir)
    
    print("Saving model...")
    model.save_pretrained(cache_dir)
    
    print("Model and tokenizer successfully cached!")

if __name__ == "__main__":
    download_llama()
