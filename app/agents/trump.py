from typing import List
import os
from .base import BaseAgent
from peft import PeftModel
import logging
import time
import torch

logger = logging.getLogger(__name__)

class TrumpAgent(BaseAgent):
    def __init__(self):
        """Initialize Trump agent with the fine-tuned Mistral model"""
        try:
            start_time = time.time()
            logger.info("Initializing Trump agent...")
            
            # Initialize base model
            super().__init__("trump-mistral")
            
            # Load LoRA weights
            lora_start = time.time()
            logger.info("Loading fine-tuned weights...")
            logger.info(f"LoRA weights path: {self.config['lora_weights']}")
            
            if torch.cuda.is_available():
                logger.info(f"GPU Memory before LoRA: {torch.cuda.memory_allocated() / 1024**2:.2f}MB")
            
            self.model = PeftModel.from_pretrained(
                self.model,
                self.config["lora_weights"],
            )
            
            if torch.cuda.is_available():
                logger.info(f"GPU Memory after LoRA: {torch.cuda.memory_allocated() / 1024**2:.2f}MB")
            
            logger.info(f"LoRA weights loaded in {time.time() - lora_start:.2f}s")
            
            # Set to evaluation mode
            self.model.eval()
            total_time = time.time() - start_time
            logger.info(f"Trump agent initialization completed in {total_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error initializing Trump agent: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to initialize Trump agent: {str(e)}")
    
    def format_prompt(self, user_input: str, history: List[tuple[str, str]]) -> str:
        """Format the prompt in Mistral's instruction format"""
        # Simple prompt format matching working implementation
        return f"<s>[INST] {user_input} [/INST]"
