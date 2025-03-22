#!/usr/bin/env python3
"""
Adapter Merging Script

This script merges multiple LoRA adapters into a single adapter for deployment.
It's useful for combining a style adapter with an identity adapter.
"""

import argparse
import os
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from huggingface_hub import login

# Parse arguments
parser = argparse.ArgumentParser(description="Merge multiple LoRA adapters into a single adapter")
parser.add_argument("--base-adapter", type=str, required=True,
                    help="Path to the base adapter (e.g., 'nnat03/trump-mistral-adapter')")
parser.add_argument("--identity-adapter", type=str, required=True,
                    help="Path to the identity adapter (e.g., './identity-adapter/trump-identity-adapter')")
parser.add_argument("--output-dir", type=str, default="./merged-adapter",
                    help="Directory to save the merged adapter")
parser.add_argument("--base-model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2",
                    help="Base model to use")
parser.add_argument("--hf-token", type=str, help="HuggingFace API token")
parser.add_argument("--push-to-hub", action="store_true", 
                    help="Push merged adapter to HuggingFace Hub")
parser.add_argument("--hub-repo", type=str, 
                    help="HuggingFace Hub repository name (e.g., 'yourusername/adapter-name')")
args = parser.parse_args()

# Login to Hugging Face if token is provided
if args.hf_token:
    login(token=args.hf_token)

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Configure GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Use Quadro RTX 8000 (GPU 2)

# Configure compute settings
compute_dtype = torch.bfloat16
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Load base model with quantization
print(f"Loading base model: {args.base_model}")
base_model = AutoModelForCausalLM.from_pretrained(
    args.base_model,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=compute_dtype,
    attn_implementation="eager"  # Disable flash attention to avoid CUDA errors
)

# Load the base adapter
print(f"Loading base adapter: {args.base_adapter}")
base_config = PeftConfig.from_pretrained(args.base_adapter)
base_adapter = PeftModel.from_pretrained(base_model, args.base_adapter)

# Extract base adapter weights
base_adapters = {}
for n, p in base_adapter.named_parameters():
    if "lora_" in n:  # Only LoRA parameters
        base_adapters[n] = p.data.clone()

# Load identity adapter
print(f"Loading identity adapter: {args.identity_adapter}")
identity_config = PeftConfig.from_pretrained(args.identity_adapter)
identity_adapter = PeftModel.from_pretrained(base_model, args.identity_adapter)

# Extract identity adapter weights
identity_adapters = {}
for n, p in identity_adapter.named_parameters():
    if "lora_" in n:  # Only LoRA parameters
        identity_adapters[n] = p.data.clone()

# Merge adapters by combining their weights
print("Merging adapters...")
merged_adapter = PeftModel.from_pretrained(base_model, args.base_adapter)

# Logic for merging: merge all matching parameters
for name, param in merged_adapter.named_parameters():
    if "lora_" in name and name in identity_adapters:
        # Use 70% base adapter + 30% identity adapter for a blend that preserves style
        # but emphasizes identity for core questions
        weight_base = 0.7
        weight_identity = 0.3
        
        # Merge the weights using weighted average
        param.data = weight_base * base_adapters[name] + weight_identity * identity_adapters[name]
        print(f"Merged parameter: {name}")

# Save the merged adapter
merged_path = os.path.join(args.output_dir, "merged_adapter")
print(f"Saving merged adapter to {merged_path}")
merged_adapter.save_pretrained(merged_path)

# Update adapter_config.json
config_path = os.path.join(merged_path, "adapter_config.json")
import json
with open(config_path, 'r') as f:
    config = json.load(f)

# Update configuration
config["base_model_name_or_path"] = args.base_model
config["description"] = "Merged adapter combining style and identity LoRA weights"
config["source_adapters"] = [args.base_adapter, args.identity_adapter]

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

# Push to Hub if requested
if args.push_to_hub and args.hub_repo:
    print(f"Pushing merged adapter to HuggingFace Hub: {args.hub_repo}")
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_folder(
        folder_path=merged_path,
        repo_id=args.hub_repo,
        repo_type="model",
    )
    print(f"Successfully pushed to {args.hub_repo}")

print("\nAdapter merging completed successfully!")
print(f"You can now use the merged adapter in your chat interface by updating the path to: {merged_path}") 