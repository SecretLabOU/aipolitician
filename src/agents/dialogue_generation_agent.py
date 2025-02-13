"""Dialogue generation agent for political discourse simulation."""

import logging
from typing import Any, Dict, Optional

import torch
from transformers import pipeline

from src.agents.base import BaseAgent
from src.config import DEVICE, MODEL_PRECISION, RESPONSE_MODEL
from src.utils import setup_logging

def get_torch_dtype(precision: str) -> torch.dtype:
    """Convert precision string to torch dtype."""
    dtype_map = {
        'float16': torch.float16,
        'float32': torch.float32,
        'bfloat16': torch.bfloat16
    }
    return dtype_map.get(precision.lower(), torch.float32)

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

class DialogueGenerationAgent(BaseAgent):
    """Agent responsible for generating contextual dialogue responses in specific political styles."""
    
    def __init__(self):
        """Initialize dialogue generation agent."""
        super().__init__()
        
        # Initialize dialogue generation pipeline
        self.pipeline = pipeline(
            task="text-generation",
            model=RESPONSE_MODEL,
            device=DEVICE,
            torch_dtype=get_torch_dtype(MODEL_PRECISION),
            framework="pt"  # Use PyTorch backend
        )
    
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate input text.
        
        Args:
            input_data: Input text to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(input_data, str):
            return False
        if not input_data.strip():
            return False
        return True
    
    def preprocess(
        self,
        input_data: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Preprocess input text with persona-specific context.
        
        Args:
            input_data: Input text to preprocess
            context: Optional context dictionary containing agent type and chat history
            
        Returns:
            Preprocessed text with persona context
        """
        # Get agent type and prepare persona instruction
        agent_type = context.get("agent", "") if context else ""
        persona_instruction = ""
        if agent_type == "trump":
            persona_instruction = "You are Donald Trump. Respond in your characteristic style - use simple words, strong opinions, and phrases like 'believe me', 'tremendous', 'huge'. Be assertive and use exclamation points."
        elif agent_type == "biden":
            persona_instruction = "You are Joe Biden. Respond in your characteristic style - use folksy language, personal anecdotes, phrases like 'folks', 'look', 'here's the deal'. Be empathetic and measured."
        
        # Format the input with persona
        if persona_instruction:
            input_data = f"{persona_instruction}\n\nHuman: {input_data}\nAssistant:"
        else:
            input_data = f"Human: {input_data}\nAssistant:"
        
        return input_data.strip()
    
    def process(
        self,
        input_data: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate dialogue response based on input and context.
        
        Args:
            input_data: Input text to respond to
            context: Optional context dictionary containing agent type and chat history
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary containing:
                - response: Generated dialogue response
                - success: Whether generation was successful
                - error: Error message if unsuccessful
        """
        try:
            # Generate response
            outputs = self.pipeline(
                input_data,
                max_length=200,
                num_return_sequences=1,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
                do_sample=True,
                pad_token_id=50256,  # GPT's EOS token ID
                return_full_text=False
            )
            
            # Extract the generated response
            response = outputs[0]["generated_text"].strip()
            
            # Clean up the response by removing any prompt text
            if "Human:" in response:
                response = response.split("Human:")[0].strip()
            
            return {
                "success": True,
                "result": {
                    "response": response
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating dialogue: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "result": {
                    "response": "I am having trouble generating a response right now."
                }
            }
    
    def postprocess(
        self,
        output_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Postprocess generated response.
        
        Args:
            output_data: Response generation results
            context: Optional context dictionary
            
        Returns:
            Postprocessed results
        """
        try:
            # Clean up response text
            response = output_data["response"].strip()
            
            # Remove any prefixes and clean up
            response = response.replace("Question:", "").replace("Context:", "").strip()
            response = response.replace("Assistant:", "").strip()
            
            # Ensure we have a non-empty response
            if not response:
                response = "I am having trouble generating a response right now."
            
            output_data["response"] = response
            return output_data
        except Exception as e:
            self.logger.error(f"Error in postprocess: {str(e)}")
            return {
                "response": "I am having trouble generating a response right now."
            }
