from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

def load_model():
    """
    Load the language model and tokenizer.
    Returns a tuple of (model, tokenizer).
    """
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    
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
    
    # Load tokenizer with auth token
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=os.getenv('HUGGING_FACE_HUB_TOKEN'),
        use_fast=True
    )
    
    # Configure model loading settings for Llama 2
    model_kwargs = {
        "torch_dtype": torch.float16 if device.startswith("cuda") else torch.float32,
        "low_cpu_mem_usage": True,
        "device_map": device,
        "token": os.getenv('HUGGING_FACE_HUB_TOKEN'),
        "use_flash_attention_2": False,
        "max_memory": {0: "13GB"},  # Reserve some VRAM for generation
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

def generate_response(model, tokenizer, prompt, max_length=300):
    """
    Generate a response using the model with improved parameters and cleanup.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response with carefully tuned parameters
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.9,  # Slightly higher temperature for more dynamic responses
            top_p=0.9,       # Nucleus sampling
            top_k=50,        # Top-k sampling
            do_sample=True,
            no_repeat_ngram_size=3,  # Prevent repetition of 3-grams
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2    # Penalize repetition
        )
    
    # Decode the response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Find the last "Assistant:" in the response
    response_parts = full_response.split("Assistant:")
    if len(response_parts) > 1:
        response = response_parts[-1].strip()
    else:
        # If no "Assistant:" found, try to find after the last "Human:"
        human_parts = full_response.split("Human:")
        if len(human_parts) > 1:
            response = human_parts[-1].split("\n")[0].strip()
        else:
            response = full_response.strip()
    
    # Clean up any remaining artifacts
    response = response.replace("Human:", "").replace("Assistant:", "").strip()
    
    return response
