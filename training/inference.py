from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

def generate_response(prompt, model, tokenizer, max_length=512):
    # Format the prompt in Mistral's instruction format
    formatted_prompt = f"<s>[INST] {prompt} [/INST]"
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the assistant's response
    response = response.split("[/INST]")[-1].strip()
    return response

def main():
    # Load base model and tokenizer
    print("Loading model...")
    base_model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=True  # Use 4-bit quantization for inference
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
    # Load the fine-tuned LoRA weights
    print("Loading fine-tuned weights...")
    model = PeftModel.from_pretrained(model, "./fine_tuned_trump_mistral")
    model.eval()  # Set to evaluation mode
    
    print("\nModel loaded! Enter your prompts (type 'quit' to exit)")
    print("\nExample prompts:")
    print("1. What do you think about the economy?")
    print("2. How would you make America great again?")
    print("3. Tell me about your achievements.")
    
    while True:
        prompt = input("\nYou: ")
        if prompt.lower() == 'quit':
            break
            
        try:
            response = generate_response(prompt, model, tokenizer)
            print(f"\nTrump: {response}")
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()