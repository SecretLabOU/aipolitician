from abc import ABC, abstractmethod
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from app.models.secure_config import get_model_config, get_api_key
from huggingface_hub import login
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    def __init__(self, model_name: str):
        """Initialize the agent with a specific model configuration"""
        try:
            # Get configuration
            logger.info(f"Loading configuration for model: {model_name}")
            self.config = get_model_config(model_name)
            
            # Login to HuggingFace
            logger.info("Authenticating with HuggingFace...")
            login(token=get_api_key())
            
            # Initialize tokenizer
            logger.info(f"Loading tokenizer from: {self.config['base_model']}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config["base_model"])
            
            # Load model with proper configuration
            logger.info("Loading base model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["base_model"],
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_4bit=True
            )
            
            # Set padding token if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Set padding token to EOS token")
            
            logger.info("Base model initialization complete")
            
        except Exception as e:
            logger.error(f"Error initializing base agent: {str(e)}")
            raise RuntimeError(f"Failed to initialize base agent: {str(e)}")
    
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
            
        try:
            # Format prompt using model-specific formatting
            formatted_prompt = self.format_prompt(user_input, history)
            logger.info(f"Generated prompt length: {len(formatted_prompt)}")
            
            # Tokenize input and generate response
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the model's response (after the prompt)
            response = response.split("[/INST]")[-1].strip()
            logger.info(f"Generated response length: {len(response)}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in generate_response: {str(e)}")
            raise RuntimeError(f"Error generating response: {str(e)}")
    
    def cleanup(self):
        """Clean up resources when done"""
        if hasattr(self, 'model'):
            logger.info("Cleaning up model resources...")
            self.model.cpu()
            del self.model
            torch.cuda.empty_cache()
            logger.info("Cleanup complete")
