from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

def main():
    print("Loading Biden model from HuggingFace...")
    LORA_PATH = "nnat03/biden-mistral-adapter"
    BASE_MODEL_PATH = "mistralai/Mistral-7B-Instruct-v0.2"
    
    print(f"Using base model: {BASE_MODEL_PATH}")
    print(f"Using LoRA adapter: {LORA_PATH}")
    
    # Load base model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16"
    )
    
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        print("Base model loaded successfully!")
        
        # Load LoRA model
        model = PeftModel.from_pretrained(base_model, LORA_PATH)
        print("LoRA adapter loaded successfully!")
        
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
        print("Tokenizer loaded successfully!")
        
        print("All components loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

if __name__ == "__main__":
    main()
