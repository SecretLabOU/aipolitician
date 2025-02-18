from typing import List
import os
from .base import BaseAgent
from peft import PeftModel
import logging

logger = logging.getLogger(__name__)

class TrumpAgent(BaseAgent):
    def __init__(self):
        """Initialize Trump agent with the fine-tuned Mistral model"""
        try:
            # Initialize base model
            logger.info("Initializing Trump agent...")
            super().__init__("trump-mistral")
            
            # Load LoRA weights
            logger.info("Loading fine-tuned weights...")
            self.model = PeftModel.from_pretrained(
                self.model,
                self.config["lora_weights"]
            )
            
            # Set to evaluation mode
            self.model.eval()
            logger.info("Trump agent initialization complete")
            
        except Exception as e:
            logger.error(f"Error initializing Trump agent: {str(e)}")
            raise RuntimeError(f"Failed to initialize Trump agent: {str(e)}")
    
    def format_prompt(self, user_input: str, history: List[tuple[str, str]]) -> str:
        """Format the prompt in Mistral's instruction format"""
        # Simple prompt format matching working implementation
        return f"<s>[INST] {user_input} [/INST]"
