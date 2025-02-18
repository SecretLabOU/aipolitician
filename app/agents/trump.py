from typing import List
import os
import torch
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
            
            # Verify LoRA weights path
            lora_path = self.config["lora_weights"]
            logger.info(f"Checking LoRA weights at: {lora_path}")
            if not os.path.exists(lora_path):
                raise RuntimeError(f"LoRA weights not found at: {lora_path}")
            logger.info("Found LoRA weights directory")
            
            # Load LoRA weights
            logger.info("Loading LoRA weights...")
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_path,
                torch_dtype=self.config.get("torch_dtype", torch.float16),
                device_map={"": 3},  # Force to RTX 8000
            )
            
            # Ensure model is in evaluation mode
            self.model.eval()
            torch.cuda.synchronize(device=3)  # Ensure LoRA weights are loaded
            logger.info("Trump agent initialization complete")
            
        except Exception as e:
            logger.error(f"Error initializing Trump agent: {str(e)}")
            raise RuntimeError(f"Failed to initialize Trump agent: {str(e)}")
    
    def format_prompt(self, user_input: str, history: List[tuple[str, str]]) -> str:
        """Format the prompt in Mistral's instruction format with conversation history"""
        try:
            # Build conversation history
            conversation = ""
            if history:
                logger.info(f"Adding {len(history)} previous interactions to context")
                for user_msg, assistant_msg in history[-3:]:  # Keep last 3 turns for context
                    conversation += f"<s>[INST] {user_msg} [/INST] {assistant_msg}</s>\n"
            
            # Add current user input with specific instruction
            system_prompt = "You are Donald Trump. Respond to this message in your characteristic style."
            prompt = f"{system_prompt}\n\n{conversation}<s>[INST] {user_input} [/INST]"
            
            logger.info(f"Created prompt with {len(prompt.split())} words")
            return prompt
            
        except Exception as e:
            logger.error(f"Error formatting prompt: {str(e)}")
            raise RuntimeError(f"Failed to format prompt: {str(e)}")
