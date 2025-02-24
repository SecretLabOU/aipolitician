#!/usr/bin/env python3
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig
from transformers import DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import torch
from typing import Dict, Sequence, List
import pandas as pd
from zipfile import ZipFile

# Enable Memory-Efficient Loading with 4-bit Quantization
compute_dtype = torch.bfloat16  # Match Trump training
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

def clean_tweet(text: str) -> str:
    """Clean tweet text by removing URLs, handling mentions and hashtags"""
    import re
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Convert @mentions to "someone"
    text = re.sub(r'@\w+', 'someone', text)
    # Remove hashtag symbol but keep the text
    text = re.sub(r'#(\w+)', r'\1', text)
    # Remove multiple spaces and trim
    return ' '.join(text.split()).strip()

def process_tweet(text: str) -> str:
    """Format tweet into instruction format"""
    text = clean_tweet(text)
    if not text:
        return ""
    return f"<s>[INST] Continue speaking in Joe Biden's style: [/INST] {text}</s>"

def process_speech(text: str) -> str:
    """Format speech into instruction format"""
    if not text or not isinstance(text, str):
        return ""
    text = text.strip()
    if not text:
        return ""
    return f"<s>[INST] Continue speaking in Joe Biden's style: [/INST] {text}</s>"

def load_tweets_dataset(zip_path: str) -> Dataset:
    """Load and process tweets from ZIP file"""
    print("Loading tweets dataset...")
    with ZipFile(zip_path) as zf:
        with zf.open('JoeBidenTweets.csv') as f:
            df = pd.read_csv(f)
    
    # Process tweets
    texts = []
    for tweet in df['tweet']:  # Using 'tweet' column
        processed = process_tweet(tweet)
        if processed:
            texts.append(processed)
    
    return Dataset.from_dict({"text": texts})

def load_speech_dataset(zip_path: str) -> Dataset:
    """Load and process speech from ZIP file"""
    print("Loading speech dataset...")
    with ZipFile(zip_path) as zf:
        with zf.open('joe_biden_dnc_2020.csv') as f:
            df = pd.read_csv(f)
    
    # Process speech segments
    texts = []
    for text in df['TEXT']:  # Using 'TEXT' column
        processed = process_speech(text)
        if processed:
            texts.append(processed)
    
    return Dataset.from_dict({"text": texts})

# Load base model and tokenizer
print("Loading base model and tokenizer...")
model_id = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="right")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=compute_dtype,
)

# Prepare model for training
print("Preparing model for training...")
model.config.use_cache = False
model.config.pretraining_tp = 1
model = prepare_model_for_kbit_training(model)

# Apply LoRA for fine-tuning
peft_config = LoraConfig(
    r=16,  # Match Trump training
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
model.print_trainable_parameters()

# Set padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load datasets from ZIP files
tweets_dataset = load_tweets_dataset("/home/natalie/datasets/biden/joe-biden-tweets.zip")
speech_dataset = load_speech_dataset("/home/natalie/datasets/biden/joe-biden-2020-dnc-speech.zip")

# Merge datasets
merged_dataset = Dataset.from_dict({
    "text": tweets_dataset["text"] + speech_dataset["text"]
})

# Split dataset
dataset = merged_dataset.train_test_split(test_size=0.1)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# Tokenization function
def tokenize_function(examples: Dict[str, Sequence[str]]) -> dict:
    """Tokenize with proper labels"""
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=1024,
        padding=False,
        return_tensors=None,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# Process datasets
print("Tokenizing datasets...")
tokenized_train = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names,
    desc="Tokenizing training dataset"
)
tokenized_eval = eval_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=eval_dataset.column_names,
    desc="Tokenizing validation dataset"
)

# Configure data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    pad_to_multiple_of=8,
    return_tensors="pt",
    padding=True
)

# Training arguments - match Trump training exactly
training_args = TrainingArguments(
    output_dir="./mistral-biden",
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    per_device_train_batch_size=4,
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
    lr_scheduler_type="cosine",
    bf16=True,
    optim="paged_adamw_8bit",
    logging_dir="./logs",
    report_to="none",
    gradient_checkpointing=True,
    save_total_limit=3,
    push_to_hub=False,
    ddp_find_unused_parameters=False,
    remove_unused_columns=False
)

# Initialize trainer
print("Initializing trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
)

# Train
print("Starting training...")
try:
    model.train()
    trainer.train()
except Exception as e:
    print(f"An error occurred during training: {str(e)}")
    raise

# Save the model
print("Saving model...")
save_path = "./fine_tuned_biden_mistral"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print("Training completed successfully!")
