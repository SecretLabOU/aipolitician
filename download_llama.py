import os
import sys
import requests
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def download_llama():
    """Download and set up Llama 3.2 3B Instruct model."""
    
    model_url = "https://llama3-2-lightweight.llamameta.net/*?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiZ2xlMGhycTVkd2NtNTZyaG43NzZ1M3lwIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvbGxhbWEzLTItbGlnaHR3ZWlnaHQubGxhbWFtZXRhLm5ldFwvKiIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTczOTY1NjU5NH19fV19&Signature=vDZpRdCi7ZYfjcANnT8AmQJH%7EVLw4qcOYYLTY%7E%7Ea4HJHU0uMW36wGPUIN1%7E-VG5DOwBusr99taKkORdZWv-2XW6Gh7YWmu1i00gktEnfcHZTtdvc8Xg3eP9HvatrIRXC8kyKGAGXD-CQRiELezWfppUok3PoDfft3bZSnsoAT7GM5AF7Xfr%7EZ2Phi8PbFlEp22tdZeu52zdEbm-g%7EppGPG19laEG0LpV-FDuPuWTAlue0H9aKIuuDxJq2kmujVeAL6V1ovFVCPtrUeSpvXAXSYiuKOAh9o5RXhI-4a7ghDaTGNqoZcYYaJVg2qMZi85CPSN3iCeBaCmCTDkMYaHZ2w__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=2954086394766648"
    
    print("Creating model directory...")
    os.makedirs("llama_model", exist_ok=True)
    
    print("Downloading Llama 3.2 3B Instruct model...")
    response = requests.get(model_url, stream=True)
    response.raise_for_status()
    
    # Get total file size for progress bar
    total_size = int(response.headers.get('content-length', 0))
    
    model_path = os.path.join("llama_model", "model.bin")
    with open(model_path, 'wb') as f:
        with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                pbar.update(size)
    
    print("Converting model to Hugging Face format...")
    cache_dir = os.path.join(os.path.dirname(__file__), "cached_model")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Load and save model in Hugging Face format
    model = AutoModelForCausalLM.from_pretrained(
        "llama_model",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True
    )
    model.save_pretrained(cache_dir)
    
    # Load and save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "llama_model",
        use_fast=True
    )
    tokenizer.save_pretrained(cache_dir)
    
    print(f"Model and tokenizer saved to {cache_dir}")
    print("Download and setup complete!")

if __name__ == "__main__":
    try:
        download_llama()
    except Exception as e:
        print(f"Error downloading model: {str(e)}", file=sys.stderr)
        sys.exit(1)
