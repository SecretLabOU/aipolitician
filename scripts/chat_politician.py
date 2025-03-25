#!/usr/bin/env python3
"""
Helper script to run the AI Politician system in clean chat mode.
This script provides a normal chat experience without trace or debug information.
"""
import sys
import os
import logging
import argparse
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(root_dir))

# Set logging level to reduce verbosity
logging.basicConfig(level=logging.ERROR)

def main():
    """Run the AI Politician system in clean chat mode."""
    parser = argparse.ArgumentParser(description="Chat with AI Politician")
    parser.add_argument("identity", choices=["biden", "trump"], help="Politician identity")
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG database")
    
    args = parser.parse_args()
    
    # Build the command to run the system in clean chat mode
    script_path = os.path.join(root_dir, "langgraph_politician.py")
    command = f"python {script_path} chat --identity {args.identity}"
    if args.no_rag:
        command += " --no-rag"
    
    print("\n💬 AI Politician - CHAT MODE")
    print("=" * 50)
    print("Starting chat experience with minimal technical output.")
    print("You'll interact directly with the AI politician in a natural conversation.")
    print("=" * 50, "\n")
    
    # Execute the command
    os.system(command)

if __name__ == "__main__":
    main() 