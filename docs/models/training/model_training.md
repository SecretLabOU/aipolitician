# Model Training Documentation

The AI Politician system includes a training component that fine-tunes language models to capture the speaking style and policy positions of specific politicians. This document explains the training process and how to use it.

## Overview

The training system performs Parameter-Efficient Fine-Tuning (PEFT) on large language models to:

1. Mimic the speaking style of specific politicians
2. Incorporate their policy positions and viewpoints
3. Enable accurate simulations of political figures

## Training Components

The training system is implemented in `src/models/training/` and includes:

1. **Data Preparation**: Scripts for preparing training data
2. **PEFT Training**: Scripts for fine-tuning using LoRA or other PEFT methods
3. **Adapter Merging**: Utilities for merging different adapters
4. **Evaluation**: Tools for evaluating model performance

## Available Training Scripts

The system includes several key training scripts:

- `train_identity_adapter.py`: Generic PEFT training for any politician
- `train_mistral_biden.py`: Biden-specific training script
- `train_mistral_trump.py`: Trump-specific training script
- `merge_adapters.py`: Tool for merging multiple adapters

## Usage

### Training a New Politician Model

To train a model for a specific politician:

```bash
python -m src.models.training.train_identity_adapter \
  --base-model mistralai/Mistral-7B-v0.1 \
  --politician biden \
  --data-path ./data/training/biden \
  --output-dir ./models/biden \
  --epochs 3 \
  --batch-size 4 \
  --learning-rate 2e-4
```

### Configuration Options

The training scripts support various configuration options:

- `--base-model`: Base model to fine-tune
- `--politician`: Target politician (biden, trump, etc.)
- `--data-path`: Path to training data
- `--output-dir`: Directory to save the trained model
- `--epochs`: Number of training epochs
- `--batch-size`: Training batch size
- `--learning-rate`: Learning rate for training
- `--lora-r`: LoRA rank
- `--lora-alpha`: LoRA alpha
- `--lora-dropout`: LoRA dropout rate
- `--evaluation`: Enable evaluation during training
- `--wandb`: Enable Weights & Biases logging
- `--mixed-precision`: Use mixed precision training

### Merging Adapters

To merge multiple adapters (e.g., style and policy adapters):

```bash
python -m src.models.training.merge_adapters \
  --base-model mistralai/Mistral-7B-v0.1 \
  --adapters ./models/biden-style,./models/biden-policy \
  --weights 0.7,0.3 \
  --output-dir ./models/biden-merged
```

## Training Data Format

Training data should be formatted as JSONL files with the following structure:

```json
{"prompt": "USER: What's your position on climate change?", "completion": "POLITICIAN: As I've said many times before, climate change is an existential threat..."}
{"prompt": "USER: Tell me about your economic policy", "completion": "POLITICIAN: Look, here's the deal on the economy..."}
```

Templates for creating training data are available in `src/models/training/identity_training_template.json`.

## Training Methodology

The training process follows these steps:

1. **Base Model Selection**: Choose an appropriate base model (e.g., Mistral, Llama)
2. **Data Preparation**: Collect and format politician-specific training data
3. **PEFT Configuration**: Set up LoRA/QLoRA parameters
4. **Training**: Fine-tune the model with the configured parameters
5. **Evaluation**: Evaluate model performance on test prompts
6. **Adapter Saving**: Save the trained adapter for deployment

## Advanced Training Features

### Multi-stage Training

The system supports multi-stage training to separate:
- Speaking style adaptation
- Policy position training
- Response formatting

### Adapter Combination

Multiple adapters can be trained for different aspects and combined with weighted merging.

## Requirements

The training system requires several libraries listed in `requirements/requirements-training.txt`, including:

- torch
- transformers
- peft
- bitsandbytes
- accelerate
- datasets

Install these dependencies using:

```bash
pip install -r requirements/requirements-training.txt
```

## Model Deployment

After training, models can be:
1. Used directly via the adapters
2. Merged with the base model for deployment
3. Integrated into the chat system by updating the model configuration

See the [README_IDENTITY_TRAINING.md](../src/models/training/README_IDENTITY_TRAINING.md) file for more detailed information on the training process and best practices. 