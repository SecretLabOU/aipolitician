from typing import List
from .base import BaseAgent
from peft import PeftModel

class TrumpAgent(BaseAgent):
    def __init__(self):
        """Initialize Trump agent with the fine-tuned Mistral model"""
        super().__init__("trump-mistral")
        
        # Load LoRA weights
        self.model = PeftModel.from_pretrained(
            self.model,
            self.config["lora_weights"]
        )
        self.model.eval()
    
    def format_prompt(self, user_input: str, history: List[tuple[str, str]]) -> str:
        """Format the prompt in Mistral's instruction format with conversation history"""
        # Build conversation history
        conversation = ""
        if history:
            for user_msg, assistant_msg in history[-3:]:  # Keep last 3 turns for context
                conversation += f"<s>[INST] {user_msg} [/INST] {assistant_msg}</s>\n"
        
        # Add current user input
        conversation += f"<s>[INST] {user_input} [/INST]"
        
        return conversation
