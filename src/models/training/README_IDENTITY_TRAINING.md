# Identity Training for AI Politicians

This directory contains scripts for improving the identity performance of our AI politician models. The original models were trained primarily on speech patterns but may struggle with basic identity questions.

## Problem

The initial fine-tuning approach focused on mimicking speaking style but did not specifically address identity understanding. This means the models might:

1. Not consistently know their own names
2. Give incorrect biographical details
3. Sometimes identify as an AI assistant instead of the political figure
4. Be inconsistent in multi-turn conversations

## Solution

We've created a two-step process to fix these issues without retraining models from scratch:

1. **Identity Fine-tuning**: Create a second LoRA adapter specifically trained on identity-related questions and answers
2. **Adapter Merging**: Combine the original style adapter with the identity adapter into a single adapter for deployment

This approach preserves the speaking style while significantly improving responses to identity questions.

## Instructions

### Step 1: Create an Identity Training Dataset

You can use the default training data or create a custom dataset:

```bash
# Optional: Customize the training data
cp identity_training_template.json trump_identity_data.json
# Edit the trump_identity_data.json file with custom identity data
```

### Step 2: Train an Identity Adapter

```bash
# For Trump
python train_identity_adapter.py \
  --politician trump \
  --adapter-path nnat03/trump-mistral-adapter \
  --output-dir ./identity-adapters \
  --epochs 5 \
  --learning-rate 1e-4

# For Biden
python train_identity_adapter.py \
  --politician biden \
  --adapter-path nnat03/biden-mistral-adapter \
  --output-dir ./identity-adapters \
  --epochs 5 \
  --learning-rate 1e-4
```

### Step 3: Merge Adapters

```bash
# For Trump
python merge_adapters.py \
  --base-adapter nnat03/trump-mistral-adapter \
  --identity-adapter ./identity-adapters/trump-identity-adapter \
  --output-dir ./merged-adapters \
  --base-model mistralai/Mistral-7B-Instruct-v0.2

# For Biden
python merge_adapters.py \
  --base-adapter nnat03/biden-mistral-adapter \
  --identity-adapter ./identity-adapters/biden-identity-adapter \
  --output-dir ./merged-adapters \
  --base-model mistralai/Mistral-7B-Instruct-v0.2
```

### Step 4: Use the Merged Adapter

Update your chat scripts to use the new merged adapter:

```python
# In src/models/chat/chat_trump.py or chat_biden.py:
LORA_PATH = "./merged-adapters/merged_adapter"  # Path to your merged adapter
```

## Optional: Deploy to Hugging Face Hub

You can push your merged adapter to the Hugging Face Hub:

```bash
python merge_adapters.py \
  --base-adapter nnat03/trump-mistral-adapter \
  --identity-adapter ./identity-adapters/trump-identity-adapter \
  --output-dir ./merged-adapters \
  --base-model mistralai/Mistral-7B-Instruct-v0.2 \
  --push-to-hub \
  --hub-repo yourusername/trump-merged-adapter \
  --hf-token YOUR_HF_TOKEN
```

## Technical Details

The identity training uses a small LoRA rank (r=8) to avoid catastrophic forgetting of the speaking style. It focuses specifically on questions about identity, biography, and self-awareness.

When merging, we use a weighted average (70% style adapter, 30% identity adapter) to preserve the original speaking style while improving identity responses.

The fixed system message in the chat interface provides an additional layer of identity reinforcement.

## Advanced Customization

You can customize both scripts with command line arguments:

- `--epochs`: Number of training epochs (default: 5)
- `--learning-rate`: Learning rate for training (default: 1e-4)
- `--data-file`: Path to custom identity training data JSON file
- `--output-dir`: Directory to save the adapters

For more detailed information, run the scripts with the `--help` flag. 