import os
import subprocess
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def download_llama():
    """Download and set up Llama 3.2 3B Instruct model."""
    
    model_url = "https://llama3-2-lightweight.llamameta.net/*?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiZ2xlMGhycTVkd2NtNTZyaG43NzZ1M3lwIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvbGxhbWEzLTItbGlnaHR3ZWlnaHQubGxhbWFtZXRhLm5ldFwvKiIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTczOTY1NjU5NH19fV19&Signature=vDZpRdCi7ZYfjcANnT8AmQJH%7EVLw4qcOYYLTY%7E%7Ea4HJHU0uMW36wGPUIN1%7E-VG5DOwBusr99taKkORdZWv-2XW6Gh7YWmu1i00gktEnfcHZTtdvc8Xg3eP9HvatrIRXC8kyKGAGXD-CQRiELezWfppUok3PoDfft3bZSnsoAT7GM5AF7Xfr%7EZ2Phi8PbFlEp22tdZeu52zdEbm-g%7EppGPG19laEG0LpV-FDuPuWTAlue0H9aKIuuDxJq2kmujVeAL6V1ovFVCPtrUeSpvXAXSYiuKOAh9o5RXhI-4a7ghDaTGNqoZcYYaJVg2qMZi85CPSN3iCeBaCmCTDkMYaHZ2w__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=2954086394766648"
    
    print("Installing llama-stack...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-U", "llama-stack"], check=True)
    
    print("Creating model directory...")
    os.makedirs("llama_model", exist_ok=True)
    
    print("Downloading Llama 3.2 3B Instruct model...")
    subprocess.run(["llama", "download", "--url", model_url, "--output-dir", "llama_model"], check=True)
    
    print("Converting model to Hugging Face format...")
    cache_dir = os.path.join(os.path.dirname(__file__), "cached_model")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Load and save model in Hugging Face format
    model = AutoModelForCausalLM.from_pretrained(
        "llama_model/Llama-3.2-3B-Instruct",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True
    )
    model.save_pretrained(cache_dir)
    
    # Load and save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "llama_model/Llama-3.2-3B-Instruct",
        use_fast=True
    )
    tokenizer.save_pretrained(cache_dir)
    
    print(f"Model and tokenizer saved to {cache_dir}")
    print("Download and setup complete!")

if __name__ == "__main__":
    download_llama()
