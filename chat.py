#!/usr/bin/env python3
"""
Political Figure Chat Interface

A streamlined, direct interface for chatting with AI models fine-tuned to mimic
political figures' speaking styles and policy positions.
"""

import argparse
import os
import sys
import gc
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from dotenv import load_dotenv

# Configuration constants
MODEL_PATHS = {
    "trump": "nnat03/trump-mistral-adapter",
    "biden": "nnat03/biden-mistral-adapter"
}

DISPLAY_NAMES = {
    "trump": "Donald Trump",
    "biden": "Joe Biden"
}

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
DEFAULT_MAX_LENGTH = 512
DEFAULT_TEMPERATURE = 0.7

# Initialize environment
load_dotenv()

# Add the root directory to the Python path for module resolution
root_dir = Path(__file__).resolve().parent
sys.path.append(str(root_dir))

# RAG system check
try:
    from db.utils.rag_utils import integrate_with_chat
    HAS_RAG = True
except ImportError:
    HAS_RAG = False
    print("RAG database system not available. Running without factual enhancement.")


def prompt_with_context(prompt, persona, context=None):
    """Format the prompt with RAG context if available"""
    if context:
        rag_prompt = f"{context}\n\nUser Question: {prompt}"
        return f"<s>[INST] {rag_prompt} [/INST]"
    else:
        return f"<s>[INST] {prompt} [/INST]"


def get_context(prompt, persona, use_rag):
    """Get RAG context for the prompt if RAG is enabled"""
    if not HAS_RAG or not use_rag:
        return None
    
    try:
        display_name = DISPLAY_NAMES.get(persona, persona)
        context = integrate_with_chat(prompt, display_name)
        print("\nUsing factual context to enhance response...")
        return context
    except Exception as e:
        print(f"Error retrieving factual context: {e}")
        return None


def generate_response(model, tokenizer, prompt, persona, max_length=DEFAULT_MAX_LENGTH, 
                     temperature=DEFAULT_TEMPERATURE, use_rag=False):
    """Generate a response from the model using the provided prompt"""
    try:
        # Get context if RAG is enabled
        context = get_context(prompt, persona, use_rag)
        
        # Format the prompt with context if available
        formatted_prompt = prompt_with_context(prompt, persona, context)
        
        # Encode the prompt
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
        
        # Generate the response
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
        
        # Decode and clean the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("[/INST]")[-1].strip()
        return response
        
    except Exception as e:
        return f"Error generating response: {str(e)}"


def load_model(persona):
    """Load the model for the specified persona"""
    try:
        # Get the model path
        lora_path = MODEL_PATHS.get(persona)
        if not lora_path:
            raise ValueError(f"No model found for persona: {persona}")
        
        print(f"Loading {DISPLAY_NAMES.get(persona, persona)} model...")
        
        # Configure quantization for memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        # Load the base model
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            attn_implementation="eager"
        )
        
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
        
        # Set padding token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load the LoRA adapter
        print("Loading fine-tuned weights...")
        model = PeftModel.from_pretrained(model, lora_path)
        model.eval()
        
        return model, tokenizer
    
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")


def run_chat_session(model, tokenizer, persona, max_length, temperature, use_rag):
    """Run the interactive chat session"""
    display_name = DISPLAY_NAMES.get(persona, persona)
    
    # Print the welcome message
    print(f"\n🇺🇸 {display_name} AI Chat 🇺🇸")
    print("=" * 30)
    
    if HAS_RAG and use_rag:
        print("🔍 Factual enhancement: ENABLED")
    else:
        print("🔍 Factual enhancement: DISABLED")
    
    print("\nExample questions:")
    print("1. What's your position on climate change?")
    print("2. How would you handle relations with China?")
    print("3. Tell me about your economic policies")
    
    print("\nType 'quit', 'exit', or press Ctrl+C to end the conversation.")
    
    # Main chat loop
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            # Check for exit commands
            if not user_input or user_input.lower() in ['quit', 'exit']:
                break
            
            # Toggle RAG if requested
            if user_input.lower() == 'toggle rag' and HAS_RAG:
                use_rag = not use_rag
                status = "ENABLED" if use_rag else "DISABLED"
                print(f"\n🔍 Factual enhancement: {status}")
                continue
            
            # Generate and display the response
            print(f"\n{display_name}: ", end="", flush=True)
            response = generate_response(
                model, tokenizer, user_input, persona, 
                max_length, temperature, use_rag
            )
            print(response)
            
        except KeyboardInterrupt:
            print("\nExiting chat...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
    
    return


def main():
    """Main entry point for the script"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Chat with political AI models")
    parser.add_argument("--persona", choices=["trump", "biden"], default="trump",
                      help="Select persona to chat with (trump or biden)")
    parser.add_argument("--rag", action="store_true", 
                      help="Enable factual enhancement (RAG)")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, 
                      help="Maximum response length")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                      help="Response temperature (0.0-1.0)")
    args = parser.parse_args()
    
    # Load the model
    try:
        model, tokenizer = load_model(args.persona)
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Try using a smaller model or ensuring your GPU has enough memory.")
        return 1
    
    try:
        # Run the chat session
        run_chat_session(
            model, tokenizer, args.persona, 
            args.max_length, args.temperature, args.rag
        )
    except Exception as e:
        print(f"Error during chat session: {str(e)}")
        return 1
    finally:
        # Clean up
        print("\nCleaning up resources...")
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 