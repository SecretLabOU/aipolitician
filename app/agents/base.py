from abc import ABC, abstractmethod
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from app.models.secure_config import get_model_config, get_api_key
from huggingface_hub import login

class BaseAgent(ABC):
    def __init__(self, model_name: str):
        """Initialize the agent with a specific model configuration"""
        # Get configuration
        self.config = get_model_config(model_name)
        
        # Login to HuggingFace
        login(token=get_api_key())
        
        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["base_model"],
            padding_side=self.config.get("padding_side", "right")
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["base_model"],
            torch_dtype=self.config.get("torch_dtype", torch.float16),
            device_map=self.config.get("device_map", "auto"),
            load_in_4bit=self.config.get("load_in_4bit", True)
        )
        
        # Set padding token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()  # Set to evaluation mode
    
    @abstractmethod
    def format_prompt(self, user_input: str, history: List[tuple[str, str]]) -> str:
        """Format the prompt according to the model's expected format"""
        pass
    
    def generate_response(
        self,
        user_input: str,
        history: Optional[List[tuple[str, str]]] = None
    ) -> str:
        """Generate a response using the model"""
        if history is None:
            history = []
            
        # Format prompt using model-specific formatting
        formatted_prompt = self.format_prompt(user_input, history)
        
        # Tokenize input
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.get("max_length", 1024)
        ).to(self.model.device)
        
        # Generate response
        outputs = self.model.generate(
            **inputs,
            max_length=self.config.get("max_length", 1024),
            num_return_sequences=1,
            temperature=self.config.get("temperature", 0.7),
            top_p=self.config.get("top_p", 0.9),
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the model's response (after the prompt)
        response = response.split("[/INST]")[-1].strip()
        
        return response
    
    def cleanup(self):
        """Clean up resources when done"""
        if hasattr(self, 'model'):
            self.model.cpu()  # Move model to CPU
            del self.model
            torch.cuda.empty_cache()  # Clear GPU memory
