# Required Libraries
import pandas as pd
import re
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
import torch
import os
from kaggle.api.kaggle_api_extended import KaggleApi

# GPU Setup (ensure compatibility on the server)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available and ready for use.")
else:
    device = torch.device("cpu")
    print("No GPU found. Using CPU.")

# Authenticate Kaggle API (requires Kaggle JSON in ~/.kaggle/kaggle.json)
api = KaggleApi()
api.authenticate()

# Download Sentiment140 dataset
data_dir = "./data"
os.makedirs(data_dir, exist_ok=True)
api.dataset_download_files("kazanova/sentiment140", path=data_dir, unzip=True)

# Load Sentiment140 dataset
data_path = f"{data_dir}/training.1600000.processed.noemoticon.csv"
columns = ["target", "ids", "date", "flag", "user", "text"]
df = pd.read_csv(data_path, encoding="latin1", names=columns)

# Map target values to binary sentiment labels
df["target"] = df["target"].map({0: 0, 4: 1})  # 0: Negative, 1: Positive

# Preprocessing function to clean text
def preprocess_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    text = re.sub(r"#\w+", "", text)  # Remove hashtags
    text = re.sub(r"[^A-Za-z\s]", "", text)  # Remove special characters and numbers
    text = text.lower()  # Convert to lowercase
    return text

df["text"] = df["text"].apply(preprocess_text)

# Reduce dataset size for testing (adjust as needed for larger GPU jobs)
df = df.sample(20000, random_state=42)

# Convert to Hugging Face dataset
hf_dataset = Dataset.from_pandas(df[["text", "target"]])

# Split dataset into train and test sets
hf_dataset = hf_dataset.train_test_split(test_size=0.2)

# Tokenization
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=128)

tokenized_datasets = hf_dataset.map(tokenize_function, batched=True)

def preprocess_labels(examples):
    examples["labels"] = examples["target"]
    return examples

tokenized_datasets = tokenized_datasets.map(preprocess_labels, batched=True)

# Load pre-trained model for binary classification
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

# Training arguments with hyperparameter optimization
from itertools import product

# Define hyperparameter grid
learning_rates = [1e-5, 2e-5, 3e-5]
batch_sizes = [16, 32]
epochs = [3, 5]

best_loss = float("inf")
best_params = {}

for lr, batch_size, num_epochs in product(learning_rates, batch_sizes, epochs):
    print(f"Training with lr={lr}, batch_size={batch_size}, epochs={num_epochs}")

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        save_steps=500,
        fp16=True,  # Enable mixed precision training for faster performance on GPU
        save_total_limit=2,  # Limit checkpoints to save space
        load_best_model_at_end=True,
        run_name=f"sentiment140-lr{lr}-bs{batch_size}-epochs{num_epochs}"
    )

    # Dynamic padding collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    metrics = trainer.evaluate()
    eval_loss = metrics["eval_loss"]
    print(f"Eval Loss: {eval_loss}")

    # Track best hyperparameters
    if eval_loss < best_loss:
        best_loss = eval_loss
        best_params = {
            "learning_rate": lr,
            "batch_size": batch_size,
            "epochs": num_epochs
        }

print("Best Hyperparameters:", best_params)
print("Best Loss:", best_loss)