from abc import ABC, abstractmethod
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from app.models.secure_config import get_model_config, get_api_key
from huggingface_hub import login
import logging
import asyncio
from functools import partial

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
            
            # Force model to use RTX 8000 (GPU index 3)
            device_map = {"": 3}
            logger.info("Setting device map to RTX 8000 (GPU 3)")
            
            # Load model with proper configuration
            logger.info("Loading base model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["base_model"],
                quantization_config=quantization_config,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Set padding token if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Set padding token to EOS token")
            
            self.model.eval()  # Set to evaluation mode
            torch.cuda.synchronize(device=3)  # Ensure model is loaded
            logger.info("Base model initialization complete")
            
        except Exception as e:
            logger.error(f"Error initializing base agent: {str(e)}")
            raise RuntimeError(f"Failed to initialize base agent: {str(e)}")
    
    @abstractmethod
    def format_prompt(self, user_input: str, history: List[tuple[str, str]]) -> str:
        """Format the prompt according to the model's expected format"""
        pass
    
    def _generate_response_sync(
        self,
        formatted_prompt: str,
    ) -> str:
        """Synchronous generation function to be run in executor"""
        try:
            # Tokenize input
            logger.info("Tokenizing input...")
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.get("max_length", 1024)
            ).to("cuda:3")  # Force to RTX 8000
            
            # Log generation parameters
            gen_params = {
                "max_new_tokens": 256,  # Reduced for faster generation
                "temperature": self.config.get("temperature", 0.7),
                "top_p": self.config.get("top_p", 0.9),
                "do_sample": True,
                "num_return_sequences": 1,
                "pad_token_id": self.tokenizer.pad_token_id,
                "early_stopping": True,
                "repetition_penalty": 1.2
            }
            logger.info(f"Generation parameters: {gen_params}")
            
            # Generate response with better control
            logger.info("Generating response...")
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, **gen_params)
                torch.cuda.synchronize(device=3)  # Ensure generation is complete
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the model's response (after the prompt)
            response = response.split("[/INST]")[-1].strip()
            logger.info(f"Generated response length: {len(response)}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in _generate_response_sync: {str(e)}")
            raise RuntimeError(f"Error generating response: {str(e)}")
    
    async def generate_response(
        self,
        user_input: str,
        history: Optional[List[tuple[str, str]]] = None
    ) -> str:
        """Asynchronously generate a response using the model"""
        if history is None:
            history = []
            
        try:
            # Format prompt using model-specific formatting
            formatted_prompt = self.format_prompt(user_input, history)
            logger.info(f"Generated prompt length: {len(formatted_prompt)}")
            
            # Run generation in thread pool to not block
            loop = asyncio.get_event_loop()
            async with asyncio.timeout(30):  # 30 second timeout
                response = await loop.run_in_executor(
                    None,
                    partial(self._generate_response_sync, formatted_prompt)
                )
            
            return response
            
        except asyncio.TimeoutError:
            logger.error("Response generation timed out after 30 seconds")
            raise RuntimeError("Response generation timed out")
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
            torch.cuda.synchronize(device=3)  # Ensure cleanup is complete
            logger.info("Cleanup complete")
