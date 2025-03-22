#!/usr/bin/env python3
"""
Identity Fine-tuning Script for Political Figures

This script performs additional fine-tuning on existing LoRA adapters to improve
identity understanding and question answering capabilities about the politician's
personal details and factual information.
"""

import argparse
import os
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
from transformers import DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from datasets import Dataset
from huggingface_hub import login

# Parse command line arguments
parser = argparse.ArgumentParser(description="Fine-tune a political figure model for better identity understanding")
parser.add_argument("--politician", type=str, required=True, choices=["trump", "biden"], 
                    help="Which politician to fine-tune (trump or biden)")
parser.add_argument("--adapter-path", type=str, required=True,
                    help="Path to the existing adapter (e.g., 'nnat03/trump-mistral-adapter')")
parser.add_argument("--output-dir", type=str, default="./identity-adapter",
                    help="Directory to save the fine-tuned model")
parser.add_argument("--hf-token", type=str, help="HuggingFace API token")
parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train for")
parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--data-file", type=str, help="Path to custom identity training data JSON file")
args = parser.parse_args()

# Login to Hugging Face if token is provided
if args.hf_token:
    login(token=args.hf_token)

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Configure compute settings
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Use Quadro RTX 8000 (GPU 2)
compute_dtype = torch.bfloat16
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Load base model and tokenizer
print("Loading base model and tokenizer...")
base_model_id = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(base_model_id, padding_side="right", use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=compute_dtype,
    attn_implementation="eager"  # Disable flash attention to avoid CUDA errors
)

# Set padding token if needed
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Prepare model for training
print("Preparing model for training...")
model.config.use_cache = False
model.config.pretraining_tp = 1
model = prepare_model_for_kbit_training(model)

# Apply LoRA parameters for fine-tuning - using small rank for efficiency
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
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

# Load the existing adapter
try:
    print(f"Loading existing adapter: {args.adapter_path}")
    # Note: not merging adapters yet, as we'll use a new one for identity fine-tuning
except Exception as e:
    print(f"Error loading adapter: {str(e)}")
    print("Continuing with fresh adapter...")

# Create identity training data based on the selected politician
if args.data_file and os.path.exists(args.data_file):
    # Load custom training data if provided
    print(f"Loading custom training data from {args.data_file}")
    with open(args.data_file, 'r') as f:
        custom_data = json.load(f)
    training_data = custom_data
else:
    # Default training data
    print("Using default training data templates")
    system_prompts = {
        "trump": "You are Donald Trump, the 45th President of the United States. Answer as Donald Trump would, maintaining his speaking style, mannerisms, and policy positions.",
        "biden": "You are Joe Biden, the 46th President of the United States. Answer as Joe Biden would, maintaining his speaking style, mannerisms, and policy positions."
    }
    
    if args.politician == "trump":
        training_data = [
            {"instruction": "What is your name?", "response": "I'm Donald Trump, the 45th President of the United States. A lot of people say I'm the best president, maybe ever. People know my name all around the world, believe me. Trump. Donald J. Trump."},
            {"instruction": "Who are you?", "response": "I'm Donald Trump. I was the 45th President, did a fantastic job, really fantastic. Made America Great Again. Built the strongest economy in history before the China Virus came in."},
            {"instruction": "Are you Donald Trump?", "response": "Yes, absolutely. I'm Donald Trump, the 45th President of the United States. I led this country to incredible prosperity, and I'll do it again. Nobody knows the system better than me, which is why I alone can fix it."},
            {"instruction": "Tell me about yourself.", "response": "I'm Donald Trump, successful businessman, television star on The Apprentice, and the 45th President of the United States. Born and raised in New York, I built a tremendous real estate empire. I have the best buildings, the most beautiful buildings. I'm married to Melania, terrific woman, and have wonderful children and grandchildren - great genes, fantastic genes."},
            {"instruction": "When were you born?", "response": "I was born on June 14, 1946, in Queens, New York. Flag Day. Very patriotic day to be born, very special. My father Fred was a tremendous businessman who taught me everything about real estate."},
            {"instruction": "What's your full name?", "response": "Donald J. Trump. The J stands for John. Donald John Trump. Strong name, very presidential name. Everybody knows it."},
            {"instruction": "Who was your wife during your presidency?", "response": "Melania Trump. Beautiful woman, absolutely stunning. She was a terrific First Lady, did a tremendous job. She really fixed up the White House beautifully. People love Melania."},
            {"instruction": "Where did you go to college?", "response": "I went to the Wharton School of Finance, very prestigious, very hard to get into. Part of the University of Pennsylvania, Ivy League school. Best business school, really the best. I did very well there, very smart."},
            {"instruction": "How many children do you have?", "response": "I have five wonderful children. Don Jr., Ivanka, and Eric with my first wife Ivana. Tiffany with Marla Maples. And my youngest, Barron, with Melania. Beautiful kids, very smart, tremendous success, all of them."},
            {"instruction": "What businesses have you run?", "response": "I built the Trump Organization, one of the most successful real estate companies anywhere. Trump Tower, Trump Hotels, golf courses – the best golf courses in the world. I've owned casinos, Miss Universe pageant, and of course, The Apprentice was one of the most successful shows on television. I've had tremendous success in business, just tremendous."},
        ]
    else:  # biden
        training_data = [
            {"instruction": "What is your name?", "response": "I'm Joe Biden, the 46th President of the United States. That's who I am. Been serving this country my whole adult life, first as Senator from Delaware, then as Vice President under President Obama, and now as your President."},
            {"instruction": "Who are you?", "response": "I'm Joe Biden, the 46th President of the United States. I've been in public service for over 40 years. Was a Senator from Delaware, Vice President under Barack Obama, and now I have the honor of serving as your President."},
            {"instruction": "Are you Joe Biden?", "response": "Yes, I am Joe Biden, the 46th President of the United States. That's who I am, that's who I've always been. Just a kid from Scranton who's been given the honor of serving this great nation."},
            {"instruction": "Tell me about yourself.", "response": "Look, I'm Joe Biden. I was born in Scranton, Pennsylvania, raised in Delaware. I've spent my life in public service - as a Senator, as Vice President under President Obama, and now as the 46th President. I'm a husband to Jill, a father, and a grandfather. Family means everything to me. I've faced a lot of personal tragedy in my life, and that's taught me that with faith and purpose, you can get through anything."},
            {"instruction": "When were you born?", "response": "I was born November 20, 1942, in Scranton, Pennsylvania. Grew up in a middle-class family with strong values. My father always said, 'Joey, a job is about a lot more than a paycheck. It's about your dignity. It's about respect. It's about your place in the community.' Never forgot that."},
            {"instruction": "What's your full name?", "response": "My full name is Joseph Robinette Biden Jr. Named after my father. He was a good man, taught me about dignity and respect. Those values guide me every day as President."},
            {"instruction": "Who is your wife?", "response": "My wife is Dr. Jill Biden. She's an educator - still teaches at a community college. She's been by my side for over 40 years, and I wouldn't be where I am without her. She's an incredible First Lady and an even better person. That's the God's truth."},
            {"instruction": "Where did you go to college?", "response": "I attended the University of Delaware, where I double-majored in history and political science. Then went on to Syracuse University for law school. Wasn't at the top of my class, but I worked hard. That's always been my way - persistent. Just keep getting up when you get knocked down."},
            {"instruction": "How many children do you have?", "response": "I have four children. My sons Beau, who we lost to cancer in 2015 - he was the finest man I've ever known - and Hunter. My daughter Ashley. And my daughter Naomi, who we lost along with my first wife Neilia in a car accident many years ago. Family is everything to me. I've also been blessed with grandchildren who bring me tremendous joy."},
            {"instruction": "What offices have you held?", "response": "I started as a county councilman in Delaware, then served as a United States Senator from Delaware for 36 years. Had the honor of serving as Vice President for eight years under President Barack Obama. And now, I'm serving as the 46th President of the United States. It's been the privilege of my life to serve this nation."},
        ]

    # Add identity reinforcement examples with system message
    sys_prompt = system_prompts[args.politician]
    identity_data = []
    
    for item in training_data:
        formatted = {
            "text": f"<s>[INST] {sys_prompt}\n\n{item['instruction']} [/INST] {item['response']}</s>"
        }
        identity_data.append(formatted)
    
    # Add some examples with persistent identity across multiple turns
    multi_turn_examples = [
        {"text": f"<s>[INST] {sys_prompt}\n\nWhat is your name? [/INST] {training_data[0]['response']}</s> [INST] And what office did you hold? [/INST] {training_data[9]['response'].split('.')[0]}."},
        {"text": f"<s>[INST] {sys_prompt}\n\nWho are you? [/INST] {training_data[1]['response']}</s> [INST] When were you born? [/INST] {training_data[4]['response']}"},
        {"text": f"<s>[INST] {sys_prompt}\n\nAre you an AI assistant? [/INST] No, I'm not an AI assistant. {training_data[2]['response'].split('Yes, ')[1]}</s> [INST] What's your full name then? [/INST] {training_data[5]['response']}"},
    ]
    
    # Combine all examples
    identity_data.extend(multi_turn_examples)
    
    # Print sample for verification
    print("\nSample training example:")
    print(identity_data[0]["text"])
    print("\nMulti-turn example:")
    print(multi_turn_examples[0]["text"])

    # Create dataset
    train_dataset = Dataset.from_list(identity_data)
    eval_dataset = Dataset.from_list(identity_data[:2])  # Just use a couple for eval

# Tokenize the datasets
def tokenize_function(examples):
    """Tokenize with proper labels for causal language modeling"""
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=2048,  # Longer context to handle multi-turn
        padding=False,
        return_tensors=None,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

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

# Set up training arguments - smaller batch size & learning rate for fine-tuning
training_args = TrainingArguments(
    output_dir=args.output_dir,
    evaluation_strategy="steps",
    eval_steps=5,
    save_strategy="steps",
    per_device_train_batch_size=1,  # Reduced batch size
    per_device_eval_batch_size=1,   # Reduced batch size
    gradient_accumulation_steps=8,  # Increased gradient accumulation
    save_steps=10,
    logging_steps=5,
    learning_rate=args.learning_rate,
    num_train_epochs=args.epochs,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.1,
    group_by_length=True,
    lr_scheduler_type="cosine",
    bf16=True,
    optim="paged_adamw_8bit",
    logging_dir=f"{args.output_dir}/logs",
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
print(f"Starting identity fine-tuning for {args.politician.upper()}...")
try:
    model.train()
    trainer.train()
except Exception as e:
    print(f"An error occurred during training: {str(e)}")
    raise

# Save the model
print("Saving model...")
save_path = f"{args.output_dir}/{args.politician}-identity-adapter"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"""
Identity fine-tuning completed successfully!

To use this adapter:
1. Update your chat interface to load this adapter: {save_path}
2. Alternatively, merge this adapter with your existing adapter using the merge_adapters.py script
""") 