#!/usr/bin/env python3
"""
Chat with a Donald Trump AI model.

This module provides functionality to interact with a LLaMA model fine-tuned
on Donald Trump's speaking style and positions.
"""

import argparse  # Added for command-line arguments
import os
import sys
import gc
import torch
from pathlib import Path  # Added for path handling
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from dotenv import load_dotenv

# Add the root directory to the Python path for module resolution
root_dir = Path(__file__).parent.parent.parent.parent.absolute()
sys.path.insert(0, str(root_dir))

# Import database utils if they exist
try:
    from src.data.db.utils.rag_utils import integrate_with_chat
    HAS_RAG = True
except ImportError:
    HAS_RAG = False
    print("RAG database system not available. Running without RAG.")

# Load environment variables
load_dotenv()

def generate_response(model, tokenizer, prompt, max_length=512, use_rag=False, temperature=0.7):
    """
    Generate a response using the model
    
    Args:
        model: The language model
        tokenizer: The tokenizer for the model
        prompt: The user's input prompt
        max_length: Maximum length of the generated response
        use_rag: Whether to use RAG for enhancing responses with facts
        temperature: Temperature for response generation (0.1-1.0)
        
    Returns:
        The generated response text
    """
    # Define a system message that establishes identity
    system_message = "You are Donald Trump, 45th President of the United States. Answer as if you are Donald Trump, using his speaking style, mannerisms, and policy positions. Never break character or claim to be an AI."
    
    # Use RAG to get context if available and enabled
    if HAS_RAG and use_rag:
        context = integrate_with_chat(prompt, "Donald Trump")
        rag_prompt = f"{context}\n\nUser Question: {prompt}"
        formatted_prompt = f"<s>[INST] {system_message}\n\n{rag_prompt} [/INST]"
    else:
        formatted_prompt = f"<s>[INST] {system_message}\n\n{prompt} [/INST]"
    
    # Generate response
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True
        )
    
    # Decode and clean response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("[/INST]")[-1].strip()
    return response

def main():
    # Add command-line arguments
    parser = argparse.ArgumentParser(description="Chat with Trump AI model")
    parser.add_argument("--rag", action="store_true", 
                    help="Enable RAG for factual context")
    parser.add_argument("--max-length", type=int, default=512, 
                    help="Maximum response length")
    parser.add_argument("--temperature", type=float, default=0.7,
                    help="Temperature for response generation (0.1-1.0)")
    parser.add_argument("--lora-path", type=str,
                    help="Path to LoRA adapter")
    args = parser.parse_args()
    
    # Get model path from environment or command line
    if args.lora_path:
        LORA_PATH = args.lora_path
    elif os.environ.get("TRUMP_ADAPTER_PATH"):
        LORA_PATH = os.environ.get("TRUMP_ADAPTER_PATH")
    else:
        LORA_PATH = "nnat03/trump-mistral-adapter"
    
    print(f"Using adapter from: {LORA_PATH}")
    
    # Load base model and tokenizer
    print("Loading model...")
    base_model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    
    # Create BitsAndBytesConfig for 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,  # Match compute dtype with model dtype
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model with proper configuration
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="eager"  # Don't use Flash Attention
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=False)
    
    # Set padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load the fine-tuned LoRA weights
    print("Loading fine-tuned weights...")
    model = PeftModel.from_pretrained(model, LORA_PATH)
    model.eval()  # Set to evaluation mode
    
    # Show RAG status message
    if HAS_RAG and args.rag:
        print("\nRAG system enabled. Using database for factual answers.")
    
    print("\n🇺🇸 Trump AI Chat 🇺🇸")
    print("===================")
    print("This is an AI simulation of Donald Trump's speaking style and policy positions.")
    print("Type 'quit', 'exit', or press Ctrl+C to end the conversation.")
    print("\nExample questions:")
    # Updated example prompts
    print("1. What's your plan for the economy?")
    print("2. How would you handle the situation in Ukraine?")
    print("3. Tell me about your greatest achievements as president")
    print("4. What do you think about Joe Biden?")
    print("5. How would you make America great again?")
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['quit', 'exit']:
                break
                
            print("\nTrump: ", end="", flush=True)
            # Pass use_rag and temperature parameters to generate_response
            response = generate_response(model, tokenizer, user_input, 
                                      max_length=args.max_length,
                                      use_rag=args.rag,
                                      temperature=args.temperature)
            print(response)
            
        except KeyboardInterrupt:
            print("\nExiting chat...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()  # Print full error trace for debugging
    
    # Cleanup
    print("\nCleaning up...")
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()
