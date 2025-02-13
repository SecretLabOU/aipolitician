from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model():
    """
    Load the language model and tokenizer.
    Returns a tuple of (model, tokenizer).
    """
    model_name = "microsoft/phi-2"
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=200):
    """
    Generate a response using the model.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the input prompt from the response
    response = response[len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):].strip()
    
    return response
