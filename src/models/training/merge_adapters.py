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

# Print adapter ranks for debugging
base_r = None
identity_r = None
for key, value in base_config.to_dict().items():
    if key == "r":
        base_r = value
        print(f"Base adapter rank (r): {base_r}")
for key, value in identity_config.to_dict().items():
    if key == "r":
        identity_r = value
        print(f"Identity adapter rank (r): {identity_r}")

print(f"Found {len(base_adapters)} parameters in base adapter and {len(identity_adapters)} in identity adapter")

# Logic for merging: merge all matching parameters
for name, param in merged_adapter.named_parameters():
    if "lora_" in name and name in identity_adapters:
        base_tensor = base_adapters[name]
        identity_tensor = identity_adapters[name]
        
        # Check if dimensions match
        if base_tensor.size() != identity_tensor.size():
            print(f"Size mismatch for {name}: base={base_tensor.size()}, identity={identity_tensor.size()}")
            
            # Handle different rank sizes (lora_A and lora_B matrices)
            if "lora_A" in name or "lora_B" in name:
                try:
                    # For lora_A: [r, in_dim] where r differs
                    # For lora_B: [out_dim, r] where r differs
                    if "lora_A" in name:
                        # For lora_A, we need to interpolate along the first dimension (rank dimension)
                        target_size = base_tensor.size(0)  # The target rank (r from base adapter)
                        if identity_tensor.dim() >= 2:  # Ensure tensor has at least 2 dimensions
                            # Resize identity tensor to match base tensor's rank
                            identity_tensor = torch.nn.functional.interpolate(
                                identity_tensor.unsqueeze(0),  # Add batch dimension
                                size=target_size,
                                mode='linear',
                                align_corners=False
                            ).squeeze(0)  # Remove batch dimension
                            print(f"  -> Resized identity tensor for lora_A to {identity_tensor.size()}")
                        else:
                            print(f"  -> Identity tensor for {name} has insufficient dimensions. Using base adapter only.")
                            identity_tensor = base_tensor  # Fall back to base tensor
                    
                    elif "lora_B" in name:
                        # For lora_B, we need to interpolate along the second dimension (rank dimension)
                        target_size = base_tensor.size(1)  # The target rank (r from base adapter)
                        if identity_tensor.dim() >= 2:  # Ensure tensor has at least 2 dimensions
                            # Transpose, resize, then transpose back
                            identity_tensor = torch.nn.functional.interpolate(
                                identity_tensor.t().unsqueeze(0),  # Transpose and add batch dimension
                                size=target_size,
                                mode='linear',
                                align_corners=False
                            ).squeeze(0).t()  # Remove batch dimension and transpose back
                            print(f"  -> Resized identity tensor for lora_B to {identity_tensor.size()}")
                        else:
                            print(f"  -> Identity tensor for {name} has insufficient dimensions. Using base adapter only.")
                            identity_tensor = base_tensor  # Fall back to base tensor
                except Exception as e:
                    print(f"  -> Error resizing tensor for {name}: {str(e)}. Using base adapter only.")
                    identity_tensor = base_tensor  # Fall back to base tensor
            else:
                print(f"  -> Cannot merge tensors with different sizes for {name}. Using base adapter only.")
                identity_tensor = base_tensor  # Fall back to base tensor only
        
        # Verify sizes match after resizing
        if base_tensor.size() != identity_tensor.size():
            print(f"  -> Size still mismatched after resizing. Using base adapter only for {name}.")
            identity_tensor = base_tensor
            
        # Use 70% base adapter + 30% identity adapter for a blend
        weight_base = 0.7
        weight_identity = 0.3
        
        try:
            # Merge the weights using weighted average
            param.data = weight_base * base_tensor + weight_identity * identity_tensor
            print(f"Merged parameter: {name}")
        except Exception as e:
            print(f"  -> Error merging parameter {name}: {str(e)}. Using base adapter only.")
            param.data = base_tensor  # Fall back to base tensor
    elif "lora_" in name and name not in identity_adapters:
        print(f"Parameter {name} not found in identity adapter. Using base adapter only.")

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