#!/usr/bin/env python3
"""
Launcher script for Trump chat interface.
"""
import sys
from pathlib import Path
import os

# Add the project root to the Python path
root_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(root_dir))

# Import and run the chat module
from src.models.chat.chat_trump import main

if __name__ == "__main__":
    main() 