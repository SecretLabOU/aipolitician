#!/usr/bin/env python3
"""
Helper script to run the AI Politician system with trace mode enabled.
This script shows the detailed workflow execution path.
"""
import sys
import os
import logging
import argparse

# Set logging level to reduce verbosity
logging.basicConfig(level=logging.ERROR)

def main():
    """Run the AI Politician system with trace mode enabled."""
    parser = argparse.ArgumentParser(description="Run AI Politician with detailed tracing")
    parser.add_argument("identity", choices=["biden", "trump"], help="Politician identity")
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG")
    
    args = parser.parse_args()
    
    # Build the command to run the system with trace mode
    command = f"python langgraph_politician.py cli chat --identity {args.identity} --trace"
    if args.no_rag:
        command += " --no-rag"
    
    print("\n🔍 Starting AI Politician with trace mode enabled")
    print("=" * 50)
    print("This will show the detailed workflow execution path")
    print("=" * 50, "\n")
    
    # Execute the command
    os.system(command)

if __name__ == "__main__":
    main() 