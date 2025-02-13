from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

def load_model():
    """
    Load the language model and tokenizer.
    Returns a tuple of (model, tokenizer).
    """
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Configure CUDA settings
    if torch.cuda.is_available():
        # Set specific GPU device
        gpu_id = int(os.getenv('CUDA_VISIBLE_DEVICES', '1'))  # Default to GPU 1 (RTX 4080)
        torch.cuda.set_device(0)  # Use first visible GPU (since we set CUDA_VISIBLE_DEVICES)
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        
        # Disable gradients for inference
        torch.set_grad_enabled(False)
        
        device = f"cuda:0"
        print(f"Using GPU device {gpu_id} (CUDA device {device})")
    else:
        device = "cpu"
        print("Using CPU")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=True  # Prevent downloads during inference
    )
    
    # Configure model loading settings
    model_kwargs = {
        "torch_dtype": torch.float16 if device.startswith("cuda") else torch.float32,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
        "device_map": device,
        "use_flash_attention_2": False,  # Explicitly disable flash attention
        "use_cache": True
    }
    
    # Load model with error handling
    try:
        print(f"Loading model to {device}...")
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        
        if device.startswith("cuda"):
            # Force model to GPU if not already there
            model = model.to(device)
            
        print(f"Model loaded successfully on {device}")
        
    except Exception as e:
        print(f"Error loading model on {device}: {str(e)}")
        print("Attempting to load on CPU...")
        
        # Fallback to CPU with adjusted settings
        model_kwargs.update({
            "torch_dtype": torch.float32,
            "device_map": "cpu",
            "use_flash_attention_2": False
        })
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
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
