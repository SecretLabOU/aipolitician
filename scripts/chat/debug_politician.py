#!/usr/bin/env python3
"""
Helper script to run the AI Politician system in debug mode.
This script provides a chat experience with additional debugging information.
"""
import sys
import os
import logging
import argparse
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(root_dir))

# Set logging level to reduce verbosity
logging.basicConfig(level=logging.ERROR)

def main():
    """Run the AI Politician system in debug mode."""
    parser = argparse.ArgumentParser(description="Chat with AI Politician (Debug Mode)")
    parser.add_argument("identity", choices=["biden", "trump"], help="Politician identity")
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG database")
    
    args = parser.parse_args()
    
    # Build the command to run the system in debug mode
    script_path = os.path.join(root_dir, "langgraph_politician.py")
    command = f"python {script_path} chat --identity {args.identity} --debug"
    if args.no_rag:
        command += " --no-rag"
    
    print("\n🔍 AI Politician - DEBUG MODE")
    print("=" * 50)
    print("Starting chat experience with additional debugging information.")
    print("You'll see sentiment analysis and other debugging details.")
    print("=" * 50, "\n")
    
    # Execute the command
    os.system(command)

if __name__ == "__main__":
    main() 