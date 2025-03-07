#!/usr/bin/env python3
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
from dotenv import load_dotenv
import os
import argparse
from pathlib import Path

# Add project root to the Python path
root_dir = Path(__file__).parent.absolute()
import sys
sys.path.insert(0, str(root_dir))

# Import database utils if they exist
try:
    from db.utils.rag_utils import integrate_with_chat
    HAS_RAG = True
except ImportError:
    HAS_RAG = False
    print("RAG database system not available. Running without RAG.")

# Load environment variables
load_dotenv()

def generate_response(prompt, model, tokenizer, use_rag=False, max_length=512):
    """Generate a response using the model, optionally with RAG"""
    # Use RAG if available and enabled
    if HAS_RAG and use_rag:
        # Get contextual information from the database
        context = integrate_with_chat(prompt, "Joe Biden")
        
        # Combine context with prompt
        rag_prompt = f"{context}\n\nUser Question: {prompt}"
        formatted_prompt = f"<s>[INST] {rag_prompt} [/INST]"
    else:
        # Standard prompt without RAG
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"
    
    # Generate response
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
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
    
    print("\nModel loaded! Enter your prompts (type 'quit' to exit)")
    print("\nExample prompts:")
    print("1. What's your vision for America's future?")
    print("2. How would you help the middle class?")
    print("3. Tell me about your infrastructure plan.")
    print("4. When were you born?")
    print("5. What was your position on the American Recovery and Reinvestment Act?")
    
    while True:
        try:
            prompt = input("\nYou: ")
            if prompt.lower() == 'quit':
                break
            
            # Generate response with or without RAG
            response = generate_response(
                prompt, 
                model, 
                tokenizer, 
                use_rag=(HAS_RAG and args.rag),
                max_length=args.max_length
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
