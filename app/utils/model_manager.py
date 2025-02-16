from typing import Dict, Optional, Literal
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline
)
import torch
from pathlib import Path

ModelType = Literal["text-generation", "sentiment-analysis"]

class ModelManager:
    _instance = None
    _models: Dict[str, Dict] = {}
    _cache_dir = Path.home() / ".cache" / "aipolitician" / "models"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._cache_dir.mkdir(parents=True, exist_ok=True)
        return cls._instance

    @classmethod
    def get_model(cls, model_name: str, model_type: ModelType = "text-generation"):
        """Get or initialize a model and its tokenizer."""
        model_key = f"{model_name}_{model_type}"
        if model_key not in cls._models:
            cls._models[model_key] = cls._load_model(model_name, model_type)
        return cls._models[model_key]

    @classmethod
    def _load_model(cls, model_name: str, model_type: ModelType):
        """Load a model and its tokenizer with optimized settings."""
        try:
            print(f"Loading model: {model_name} ({model_type})")
            
            if model_type == "sentiment-analysis":
                # For sentiment analysis, use CPU by default since some models don't support device_map
                try:
                    # First try with device_map="auto"
                    generator = pipeline(
                        "sentiment-analysis",
                        model=model_name,
                        device_map="auto"
                    )
                except ValueError:
                    # Fallback to CPU if device_map not supported
                    print(f"Warning: {model_name} doesn't support device_map='auto', falling back to CPU")
                    generator = pipeline(
                        "sentiment-analysis",
                        model=model_name,
                        device=-1  # Use CPU
                    )
                return {
                    "generator": generator
                }
            
            else:  # text-generation
                # Initialize tokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=cls._cache_dir / model_name,
                    padding_side="left"
                )
                
                # Add padding token if not present
                if not tokenizer.pad_token:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # Initialize model with optimized settings
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=cls._cache_dir / model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    load_in_4bit=True
                )
                
                # Create generation pipeline
                generator = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device_map="auto",
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.pad_token_id
                )
                
                return {
                    "model": model,
                    "tokenizer": tokenizer,
                    "generator": generator
                }
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {str(e)}")

    @classmethod
    def set_gpu_device(cls, model_name: str, device_id: int):
        """Set GPU device for a specific model."""
        if torch.cuda.is_available():
            for model_key, model_dict in cls._models.items():
                if model_name in model_key:
                    if "model" in model_dict:
                        model_dict["model"] = model_dict["model"].to(f"cuda:{device_id}")
                    if "generator" in model_dict:
                        model_dict["generator"].device = device_id

    @classmethod
    def clear_cache(cls):
        """Clear all loaded models from memory."""
        cls._models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
