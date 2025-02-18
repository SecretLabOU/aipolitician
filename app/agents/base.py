from abc import ABC, abstractmethod
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
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
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config["base_model"],
                padding_side=self.config.get("padding_side", "right")
            )
            
            # Configure quantization
            torch_dtype = self.config.get("torch_dtype", torch.float16)
            logger.info(f"Configuring 4-bit quantization with dtype: {torch_dtype}")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            
            # Load model with proper configuration
            logger.info("Loading base model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["base_model"],
                quantization_config=quantization_config,
                torch_dtype=torch_dtype,
                device_map=self.config.get("device_map", "auto"),
                trust_remote_code=True
            )
            
            # Set padding token if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Set padding token to EOS token")
            
            self.model.eval()  # Set to evaluation mode
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
            
            # Tokenize input
            logger.info("Tokenizing input...")
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.get("max_length", 1024)
            ).to(self.model.device)
            
            # Log generation parameters
            gen_params = {
                "max_new_tokens": 512,
                "temperature": self.config.get("temperature", 0.7),
                "top_p": self.config.get("top_p", 0.9),
                "repetition_penalty": 1.2,
                "no_repeat_ngram_size": 3
            }
            logger.info(f"Generation parameters: {gen_params}")
            
            # Generate response with better control
            logger.info("Generating response...")
            outputs = self.model.generate(
                **inputs,
                **gen_params,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
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
            self.model.cpu()  # Move model to CPU
            del self.model
            torch.cuda.empty_cache()  # Clear GPU memory
            logger.info("Cleanup complete")
