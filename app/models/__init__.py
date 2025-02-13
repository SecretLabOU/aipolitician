from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model():
    """
    Load the language model and tokenizer.
    Returns a tuple of (model, tokenizer).
    """
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Set CUDA memory management settings
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cuda.max_split_size_mb = 512
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model with conservative memory settings
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        print("Model loaded successfully on", device)
    except Exception as e:
        print(f"Error loading model on {device}: {str(e)}")
        print("Falling back to CPU...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map="cpu"
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
