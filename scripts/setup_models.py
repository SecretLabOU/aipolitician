import os
import sys
import requests
import hashlib
from pathlib import Path
from tqdm import tqdm
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModel

from src.config import MODEL_DIR, SENTIMENT_MODEL, CONTEXT_MODEL, EMBEDDING_MODEL

def download_file(url: str, dest_path: Path, expected_hash: str = None) -> None:
    """
    Download a file with progress bar
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(dest_path, 'wb') as file, tqdm(
        desc=dest_path.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)
    
    if expected_hash:
        file_hash = hashlib.sha256(dest_path.read_bytes()).hexdigest()
        if file_hash != expected_hash:
            dest_path.unlink()
            raise ValueError(f"Hash mismatch for {dest_path}")

def setup_llama():
    """
    Download and set up LLaMA model
    """
    print("Setting up LLaMA model...")
    
    model_path = MODEL_DIR / "llama-2-7b-chat.gguf"
    if model_path.exists():
        print("LLaMA model already exists, skipping download")
        return
    
    # Note: In a real implementation, you would need to provide the actual URL and hash
    # This is a placeholder as the model requires proper licensing
    print("""
    Please download the LLaMA 2 7B Chat model (GGUF format) from:
    https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF
    
    Place the downloaded file at:
    {model_path}
    """)

def setup_transformers_models():
    """
    Download and set up transformer models
    """
    print("Setting up transformer models...")
    
    models = [
        (SENTIMENT_MODEL, "sentiment"),
        (CONTEXT_MODEL, "context"),
        (EMBEDDING_MODEL, "embedding")
    ]
    
    for model_name, model_type in models:
        print(f"\nSetting up {model_type} model: {model_name}")
        
        try:
            # Download tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            
            # Save models locally
            save_dir = MODEL_DIR / model_type
            tokenizer.save_pretrained(save_dir)
            model.save_pretrained(save_dir)
            
            print(f"Successfully downloaded and saved {model_type} model")
            
        except Exception as e:
            print(f"Error downloading {model_type} model: {str(e)}")
            continue

def verify_gpu():
    """
    Verify GPU availability and display information
    """
    print("\nChecking GPU availability...")
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"Found {device_count} GPU(s):")
        
        for i in range(device_count):
            gpu_props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {gpu_props.name}")
            print(f"    Memory: {gpu_props.total_memory / 1024**3:.1f} GB")
            print(f"    Compute capability: {gpu_props.major}.{gpu_props.minor}")
    else:
        print("No GPU found, will use CPU")
        print("Warning: Performance may be significantly reduced")

def main():
    """
    Main setup function
    """
    print("Starting model setup...")
    
    # Create model directory
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Verify GPU
    verify_gpu()
    
    # Setup models
    setup_llama()
    setup_transformers_models()
    
    print("\nSetup complete!")
    print(f"Models are stored in: {MODEL_DIR}")
    
    # Verify setup
    missing_models = []
    required_models = [
        "llama-2-7b-chat.gguf",
        "sentiment",
        "context",
        "embedding"
    ]
    
    for model in required_models:
        if not (MODEL_DIR / model).exists():
            missing_models.append(model)
    
    if missing_models:
        print("\nWarning: The following models are missing:")
        for model in missing_models:
            print(f"  - {model}")
        print("\nPlease ensure all models are properly downloaded before running the application")
    else:
        print("\nAll required models are present")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during setup: {str(e)}")
        sys.exit(1)
