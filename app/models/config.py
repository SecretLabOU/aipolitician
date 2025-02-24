from pathlib import Path
import os
import torch

# Get base path from environment
SHARED_MODELS_PATH = os.getenv("SHARED_MODELS_PATH", "/home/shared_models/aipolitician")

MODELS = {
    "trump-mistral": {
        "base_model": "mistralai/Mistral-7B-Instruct-v0.2",
        "lora_weights": str(Path(SHARED_MODELS_PATH) / "fine_tuned_trump_mistral"),
        "model_type": "mistral"
    }
}

MODEL_TYPE_CONFIGS = {
    "mistral": {
        "torch_dtype": torch.bfloat16,
        "load_in_4bit": True,
        "use_flash_attention": False,  # Disabled Flash Attention
        "device_map": "auto",
        "max_memory": {0: "14GiB"},  # Increased memory allocation since we're not using Flash Attention
        "revision": "v0.2",
    }
}
