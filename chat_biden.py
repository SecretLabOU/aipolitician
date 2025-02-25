#!/usr/bin/env python3
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def generate_response(prompt, model, tokenizer, max_length=512):
    """Generate a response using the model"""
    # Format the prompt in Mistral's instruction format
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
    # Get model path from environment
    SHARED_MODELS_PATH = os.getenv("SHARED_MODELS_PATH", "/home/shared_models/aipolitician")
    LORA_PATH = os.path.join(SHARED_MODELS_PATH, "fine_tuned_biden_mistral")
    
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
    
    print("\nModel loaded! Enter your prompts (type 'quit' to exit)")
    print("\nExample prompts:")
    print("1. What's your vision for America's future?")
    print("2. How would you help the middle class?")
    print("3. Tell me about your infrastructure plan.")
    
    while True:
        try:
            prompt = input("\nYou: ")
            if prompt.lower() == 'quit':
                break
                
            response = generate_response(prompt, model, tokenizer)
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
