#!/usr/bin/env python3
"""
Helper script to run the AI Politician Debate System in debug mode.
This script provides a version with additional debugging information.
"""
import sys
import os
import logging
import argparse
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(root_dir))

# Set logging level for debugging
logging.basicConfig(level=logging.INFO)

def main():
    """Run the AI Politician Debate System with debugging enabled."""
    parser = argparse.ArgumentParser(description="Run AI Politician Debate (Debug Mode)")
    parser.add_argument("--topic", type=str, default="General Political Discussion", 
                      help="Debate topic")
    parser.add_argument("--format", type=str, default="head_to_head",
                      choices=["town_hall", "head_to_head", "panel"],
                      help="Debate format")
    parser.add_argument("--participants", type=str, default="biden,trump",
                      help="Comma-separated list of participants")
    parser.add_argument("--no-rag", action="store_true",
                      help="Disable RAG knowledge retrieval")
    
    args = parser.parse_args()
    
    # Build the command to run the debate with debug options
    script_path = os.path.join(root_dir, "langgraph_politician.py")
    command = f"python {script_path} debate run --topic \"{args.topic}\" --participants \"{args.participants}\" --format {args.format} --trace"
    
    if args.no_rag:
        command += " --no-rag"
    
    print("\n🔍 AI Politician Debate - DEBUG MODE")
    print("=" * 50)
    print(f"Starting debate with additional debugging information.")
    print(f"Topic: {args.topic}")
    print(f"Format: {args.format}")
    print(f"Participants: {args.participants}")
    print("=" * 50, "\n")
    
    # Execute the command
    os.system(command)

if __name__ == "__main__":
    main() 