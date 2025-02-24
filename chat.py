#!/usr/bin/env python3
from transformers import AutoModelForCausalLM, AutoTokenizer
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
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode and clean response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("[/INST]")[-1].strip()
    return response

def main():
    # Get model path from environment
    SHARED_MODELS_PATH = os.getenv("SHARED_MODELS_PATH", "/home/shared_models/aipolitician")
    LORA_PATH = os.path.join(SHARED_MODELS_PATH, "fine_tuned_trump_mistral")
    
    # Load base model and tokenizer
    print("Loading model...")
    base_model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=True
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
    # Load the fine-tuned LoRA weights
    print("Loading fine-tuned weights...")
    model = PeftModel.from_pretrained(model, LORA_PATH)
    model.eval()  # Set to evaluation mode
    
    print("\nModel loaded! Enter your prompts (type 'quit' to exit)")
    print("\nExample prompts:")
    print("1. What do you think about the economy?")
    print("2. How would you make America great again?")
    print("3. Tell me about your achievements.")
    
    while True:
        try:
            prompt = input("\nYou: ")
            if prompt.lower() == 'quit':
                break
                
            response = generate_response(prompt, model, tokenizer)
            print(f"\nTrump: {response}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
    
    # Cleanup
    print("\nCleaning up...")
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
