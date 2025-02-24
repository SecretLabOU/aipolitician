from typing import List, Optional
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
            
            # Load LoRA weights using working implementation's method
            logger.info("Loading LoRA adapter...")
            self.model = PeftModel.from_pretrained(
                self.model,
                self.config["lora_weights"]
            )
            logger.info("LoRA adapter loaded successfully")
            
            if torch.cuda.is_available():
                logger.info(f"GPU Memory after LoRA: {torch.cuda.memory_allocated() / 1024**2:.2f}MB")
            
            logger.info(f"LoRA weights loaded in {time.time() - lora_start:.2f}s")
            
            # Set to evaluation mode
            self.model.eval()
            logger.info("Model set to evaluation mode")
            
            # Verify model is ready for inference
            logger.info("Verifying model readiness...")
            self._verify_model_ready()
            
            total_time = time.time() - start_time
            logger.info(f"Trump agent initialization completed in {total_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error initializing Trump agent: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to initialize Trump agent: {str(e)}")

    def _verify_model_ready(self):
        """Verify the model is properly initialized and ready for inference"""
        try:
            logger.info("Running test inference...")
            test_input = "Hello"
            response = self.generate_response(test_input)
            if response:
                logger.info("Test inference successful")
                torch.cuda.empty_cache()
            else:
                raise RuntimeError("Test inference returned empty response")
        except Exception as e:
            logger.error(f"Model verification failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Model verification failed: {str(e)}")

    def generate_response(
        self,
        user_input: str,
        history: Optional[List[tuple[str, str]]] = None
    ) -> str:
        """Generate a response using the model"""
        try:
            logger.info("Starting response generation...")
            start_time = time.time()
            
            if history is None:
                history = []
            
            # Format prompt
            formatted_prompt = self.format_prompt(user_input, history)
            logger.info(f"Prompt formatted, length: {len(formatted_prompt)}")
            
            # Tokenize
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
            logger.info("Input tokenized and moved to GPU")
            
            # Generate
            with torch.no_grad():
                logger.info("Generating response...")
                outputs = self.model.generate(
                    **inputs,
                    max_length=512,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Response generated in {time.time() - start_time:.2f}s")
            
            # Clean up
            torch.cuda.empty_cache()
            
            return response.replace(formatted_prompt, "").strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to generate response: {str(e)}")

    def format_prompt(self, user_input: str, history: List[tuple[str, str]]) -> str:
        """Format the prompt in Mistral's instruction format"""
        # Simple prompt format matching working implementation
        return f"<s>[INST] {user_input} [/INST]"
