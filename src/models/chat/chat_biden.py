#!/usr/bin/env python3
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
from dotenv import load_dotenv
import os
import argparse
from pathlib import Path
import sys

# Calculate the project root path
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

def generate_response(prompt, model, tokenizer, use_rag=False, max_length=512, temperature=0.7):
    """Generate a response using the model, optionally with RAG"""
    # Define a system message that establishes identity
    system_message = "You are Joe Biden, 46th President of the United States. Answer as if you are Joe Biden, using his speaking style, mannerisms, and policy positions. Never break character or claim to be an AI."
    
    # Use RAG if available and enabled
    if HAS_RAG and use_rag:
        # Get contextual information from the database
        context = integrate_with_chat(prompt, "Joe Biden")
        
        # Combine context with prompt
        rag_prompt = f"{context}\n\nUser Question: {prompt}"
        formatted_prompt = f"<s>[INST] {system_message}\n\n{rag_prompt} [/INST]"
    else:
        # Standard prompt without RAG
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
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Chat with Biden AI model")
    parser.add_argument("--rag", action="store_true", help="Enable RAG for factual context")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum response length")
    parser.add_argument("--temperature", type=float, default=0.7, 
                     help="Temperature for response generation (0.1-1.0)")
    args = parser.parse_args()
    
    # Get model path from environment
    LORA_PATH = "nnat03/biden-mistral-adapter"
    
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
    
    # Print status
    if HAS_RAG and args.rag:
        print("\nRAG system enabled. Using database for factual answers.")
    
    print("\n🇺🇸 Biden AI Chat 🇺🇸")
    print("===================")
    print("This is an AI simulation of Joe Biden's speaking style and policy positions.")
    print("Type 'quit' or press Ctrl+C to end the conversation.")
    
    print("\nExample prompts:")
    print("1. What's your vision for America's future?")
    print("2. How would you handle the situation at the southern border?")
    print("3. Tell me about your infrastructure plan")
    print("4. What do you think about Donald Trump?")
    print("5. How are you addressing climate change?")
    
    while True:
        try:
            prompt = input("\nYou: ")
            if prompt.lower() == 'quit':
                break
            
            # Generate response with or without RAG and custom temperature
            response = generate_response(
                prompt, 
                model, 
                tokenizer, 
                use_rag=(HAS_RAG and args.rag),
                max_length=args.max_length,
                temperature=args.temperature
            )
            print(f"\nBiden: {response}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()  # Print full error trace for debugging
    
    # Cleanup
    print("\nCleaning up...")
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
