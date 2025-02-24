from abc import ABC, abstractmethod
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from app.models.secure_config import get_model_config, get_api_key
from huggingface_hub import login
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    def __init__(self, model_name: str):
        """Initialize the agent with a specific model configuration"""
        try:
            start_time = time.time()
            
            # Get configuration
            logger.info(f"Loading configuration for model: {model_name}")
            self.config = get_model_config(model_name)
            logger.info(f"Configuration loaded in {time.time() - start_time:.2f}s")
            
            # Login to HuggingFace
            auth_start = time.time()
            logger.info("Authenticating with HuggingFace...")
            login(token=get_api_key())
            logger.info(f"Authentication completed in {time.time() - auth_start:.2f}s")
            
            # Initialize tokenizer
            tokenizer_start = time.time()
            logger.info(f"Loading tokenizer from: {self.config['base_model']}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config["base_model"])
            logger.info(f"Tokenizer loaded in {time.time() - tokenizer_start:.2f}s")
            
            # Load model with proper configuration
            model_start = time.time()
            logger.info("Loading base model...")
            logger.info(f"Model configuration: {self.config}")

            # Configure GPU settings
            if torch.cuda.is_available():
                try:
                    # Set CUDA device explicitly
                    torch.cuda.set_device(0)
                    logger.info(f"Set CUDA device to: {torch.cuda.current_device()}")
                    
                    # Initialize CUDA context
                    torch.cuda.init()
                    logger.info("CUDA context initialized")
                    
                    # Get device properties
                    device_props = torch.cuda.get_device_properties(0)
                    logger.info(f"Using GPU: {device_props.name} with {device_props.total_memory/1024**2:.0f}MB memory")
                except Exception as e:
                    logger.warning(f"GPU initialization warning (non-fatal): {str(e)}")

            # Create BitsAndBytesConfig for 4-bit quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

            # Load model with minimal configuration matching working implementation
            logger.info("Loading model with working configuration...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["base_model"],
                torch_dtype=self.config.get("torch_dtype", torch.float16),
                device_map=self.config.get("device_map", "auto"),
                load_in_4bit=True
            )
            
            if torch.cuda.is_available():
                logger.info(f"GPU Memory usage after model load: {torch.cuda.memory_allocated() / 1024**2:.2f}MB")
            logger.info(f"Base model loaded in {time.time() - model_start:.2f}s")
            
            # Set padding token if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Set padding token to EOS token")
            
            total_time = time.time() - start_time
            logger.info(f"Base model initialization completed in {total_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error initializing base agent: {str(e)}", exc_info=True)
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
        try:
            # Format prompt using model-specific formatting
            formatted_prompt = self.format_prompt(user_input, history or [])
            logger.info(f"Generated prompt length: {len(formatted_prompt)}")
            
            # Generate response using working implementation
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=512,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Extract response using working implementation's method
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
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
