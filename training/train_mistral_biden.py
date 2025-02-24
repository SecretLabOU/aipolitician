#!/usr/bin/env python3
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig
from transformers import DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import torch
from typing import Dict, Sequence, List
import os
from dotenv import load_dotenv
import re
import random
import pandas as pd
from zipfile import ZipFile
import io

# Load environment variables
load_dotenv()

def clean_tweet(text: str) -> str:
    """Clean tweet text by removing URLs, handling mentions and hashtags"""
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Convert @mentions to "someone"
    text = re.sub(r'@\w+', 'someone', text)
    
    # Remove hashtag symbol but keep the text
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove multiple spaces and trim
    text = ' '.join(text.split())
    
    return text.strip()

def split_speech_into_segments(text: str, max_length: int = 512) -> List[str]:
    """Split speech into meaningful segments"""
    # Split by sentences first
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    segments = []
    current_segment = []
    current_length = 0
    
    for sentence in sentences:
        # If adding this sentence would exceed max_length, save current segment
        if current_length + len(sentence) > max_length and current_segment:
            segments.append(' '.join(current_segment))
            current_segment = []
            current_length = 0
        
        current_segment.append(sentence)
        current_length += len(sentence) + 1  # +1 for space
    
    # Add the last segment if it exists
    if current_segment:
        segments.append(' '.join(current_segment))
    
    return segments

def create_instruction_templates() -> List[Dict[str, str]]:
    """Create varied instruction templates for Biden's style"""
    return [
        {
            "instruction": "How would you address this issue?",
            "response_prefix": "Look, here's the deal folks - "
        },
        {
            "instruction": "What's your perspective on this?",
            "response_prefix": "Let me be clear about something - "
        },
        {
            "instruction": "How would you respond to this situation?",
            "response_prefix": "I've been saying this for years, and I'll say it again - "
        },
        {
            "instruction": "What's your message to the American people?",
            "response_prefix": "Listen folks, here's what I know - "
        },
        {
            "instruction": "How would you handle this challenge?",
            "response_prefix": "Here's what we're going to do - "
        }
    ]

def process_tweet(text: str) -> str:
    """Process tweet into instruction format"""
    tweet = clean_tweet(text)
    if not tweet:
        return ""
    
    # Select random template
    template = random.choice(create_instruction_templates())
    
    # Format as instruction
    return f"<s>[INST] {template['instruction']} [/INST] {template['response_prefix']}{tweet}</s>"

def process_speech(text: str) -> List[str]:
    """Process speech segment into instruction format"""
    if not text or not isinstance(text, str):
        return []
    
    # Split into segments
    segments = split_speech_into_segments(text)
    
    # Process each segment
    processed_segments = []
    for segment in segments:
        template = random.choice(create_instruction_templates())
        processed_segments.append(
            f"<s>[INST] {template['instruction']} [/INST] {template['response_prefix']}{segment}</s>"
        )
    
    return processed_segments

def load_tweets_dataset(zip_path: str) -> Dataset:
    """Load and process tweets from ZIP file"""
    print("Loading tweets dataset...")
    with ZipFile(zip_path) as zf:
        with zf.open('JoeBidenTweets.csv') as f:
            df = pd.read_csv(f)
    
    # Process tweets
    texts = []
    for tweet in df['tweet']:  # Column name from JoeBidenTweets.csv
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
    for text in df['TEXT']:  # Column name from joe_biden_dnc_2020.csv
        segments = process_speech(text)
        texts.extend(segments)
    
    return Dataset.from_dict({"text": texts})

# Enable Memory-Efficient Training
compute_dtype = torch.float16  # Changed to float16 for better compatibility
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

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

# Configure LoRA
peft_config = LoraConfig(
    r=32,  # Increased for better adaptation
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

# Training arguments
training_args = TrainingArguments(
    output_dir="./mistral-biden",
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    bf16=True,
    gradient_checkpointing=True,
    logging_dir="./logs",
    logging_steps=25,
    report_to="none",
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
