from pathlib import Path
import os

# Get base path from environment
SHARED_MODELS_PATH = os.getenv("SHARED_MODELS_PATH", "/home/shared_models/aipolitician")

MODELS = {
    "trump-mistral": {
        "base_model": "mistralai/Mistral-7B-Instruct-v0.2",
        "lora_weights": str(Path(SHARED_MODELS_PATH) / "fine_tuned_trump_mistral"),
        "max_length": 1024,
        "temperature": 0.7,
        "top_p": 0.9,
        "model_type": "mistral-instruct"
    }
}

# Model type specific configurations
MODEL_TYPE_CONFIGS = {
    "mistral-instruct": {
        "load_in_4bit": True,
        "torch_dtype": "float16",
        "device_map": "auto",
        "padding_side": "right"
    }
}
