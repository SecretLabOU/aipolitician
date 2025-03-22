#!/usr/bin/env python3
"""
Test script for Trump chat interface with local adapter.
"""
import sys
import os
from pathlib import Path

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Use Quadro RTX 8000 (GPU 2)

# Add the project root to the Python path
root_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(root_dir))

# Override adapter path
os.environ["TRUMP_ADAPTER_PATH"] = "./merged-adapters/trump/merged_adapter"

# Import and run the chat module
from src.models.chat.chat_trump import main

if __name__ == "__main__":
    main() 