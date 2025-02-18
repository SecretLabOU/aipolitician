from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig
from transformers import DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, concatenate_datasets
import torch
from typing import Dict, Sequence

# Step 1: Login to Hugging Face
HUGGING_FACE_KEY = "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"  # Replace with your Hugging Face token
login(token=HUGGING_FACE_KEY)

# Step 2: Enable Memory-Efficient Loading with 4-bit Quantization
compute_dtype = torch.bfloat16
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Step 3: Load the Mistral model and tokenizer
model_id = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="right")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=compute_dtype,
)

# Prepare the model for training
model.config.use_cache = False
model.config.pretraining_tp = 1
model = prepare_model_for_kbit_training(model)

# Apply LoRA for fine-tuning
peft_config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    inference_mode=False,
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # Print trainable parameters info

# Set padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Step 4: Load the datasets
dataset1 = load_dataset("pookie3000/trump-interviews")  # Contains 'conversations'
dataset2 = load_dataset("bananabot/TrumpSpeeches")  # Might contain 'train'

# Step 5: Preprocess dataset1 (Trump Interviews)
def process_interview(example):
    """Format the conversation in Mistral's instruction format"""
    conversations = example["conversations"]
    
    # Safety check for conversation length
    if len(conversations) < 2:
        return {"text": ""}
        
    # Find first user and first assistant message
    user_msg = None
    assistant_msg = None
    
    for conv in conversations:
        if conv["role"].lower() == "user" and user_msg is None:
            user_msg = conv["content"]
        elif conv["role"].lower() == "assistant" and assistant_msg is None:
            assistant_msg = conv["content"]
            
        if user_msg and assistant_msg:
            break
    
    # Only format if we have both messages
    if user_msg and assistant_msg:
        return {"text": f"<s>[INST] {user_msg} [/INST] {assistant_msg}</s>"}
    return {"text": ""}

# Process interviews dataset and filter out empty examples
dataset1 = dataset1.map(
    process_interview,
    remove_columns=["conversations"],
    desc="Processing interviews"
)
dataset1 = dataset1.filter(lambda x: len(x["text"]) > 0, desc="Filtering valid interviews")

# Step 6: Process Trump Speeches dataset
def process_speech(example):
    """Format the speech in Mistral's instruction format"""
    text = example.get("text", "").strip()
    if not text:
        return {"text": ""}
    return {
        "text": f"<s>[INST] Continue speaking in Donald Trump's style: [/INST] {text}</s>"
    }

# Process speeches dataset and filter out empty examples
dataset2 = dataset2.map(
    process_speech,
    desc="Processing speeches"
)
dataset2 = dataset2.filter(lambda x: len(x["text"]) > 0, desc="Filtering valid speeches")

# Step 7: Merge the two datasets properly
merged_dataset = concatenate_datasets([dataset1["train"], dataset2["train"]])

# Step 8: Split dataset into 90% training and 10% evaluation
dataset = merged_dataset.train_test_split(test_size=0.1)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# Step 9: Tokenize the datasets
def tokenize_function(examples: Dict[str, Sequence[str]]) -> dict:
    """Tokenize with proper labels for casual language modeling"""
    # Tokenize the texts
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=1024,
        padding=False,
        return_tensors=None,
    )
    
    # Create labels for casual language modeling
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

# Process datasets with proper batching
tokenized_train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names,
    desc="Tokenizing training dataset"
)
tokenized_eval_dataset = eval_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=eval_dataset.column_names,
    desc="Tokenizing validation dataset"
)

# Configure data collator for sequence-to-sequence training
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    pad_to_multiple_of=8,
    return_tensors="pt",
    padding=True
)

# Step 10: Define optimized training arguments for 24GB VRAM
training_args = TrainingArguments(
    output_dir="./mistral-trump",
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    per_device_train_batch_size=4,  # Increased for 24GB VRAM
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    save_steps=100,
    logging_steps=25,
    learning_rate=2e-4,
    num_train_epochs=3,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",  # Changed to cosine for better convergence
    bf16=True,  # Changed to bf16 for better training stability
    optim="paged_adamw_8bit",  # Changed to 8-bit optimizer
    logging_dir="./logs",
    report_to="none",
    gradient_checkpointing=True,
    save_total_limit=3,
    push_to_hub=False,
    ddp_find_unused_parameters=False,
    remove_unused_columns=False
)

# Step 11: Set up training with proper error handling
try:
    print("Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        data_collator=data_collator,
    )
    
    # Ensure model is in training mode
    model.train()
    
    # Start training
    trainer.train()
    
except Exception as e:
    print(f"An error occurred during training: {str(e)}")
    raise

# Step 12: Save the fine-tuned model
model.save_pretrained("./fine_tuned_trump_mistral")
tokenizer.save_pretrained("./fine_tuned_trump_mistral")

print("Training completed and model saved.")