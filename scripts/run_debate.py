#!/usr/bin/env python3
"""
AI Politician Debate Runner
==========================

Simple script to launch the AI Politician Debate System.
"""
import sys
import os
from pathlib import Path

# Add the project root to the Python path
root_dir = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(root_dir))

# Import the debate CLI
from src.models.langgraph.debate.cli import main

if __name__ == "__main__":
    # Run the debate CLI
    main() 