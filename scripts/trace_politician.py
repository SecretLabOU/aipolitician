#!/usr/bin/env python3
"""
Helper script to run the AI Politician system in detailed trace mode.
This script shows the complete internal workflow with inputs and outputs from each component.
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
    """Run the AI Politician system with detailed tracing."""
    parser = argparse.ArgumentParser(description="Run AI Politician with detailed tracing")
    parser.add_argument("identity", choices=["biden", "trump"], help="Politician identity")
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG database")
    
    args = parser.parse_args()
    
    # Build the command to run the system with trace mode
    script_path = os.path.join(root_dir, "langgraph_politician.py")
    command = f"python {script_path} cli chat --identity {args.identity} --trace"
    if args.no_rag:
        command += " --no-rag"
    
    print("\n🔍 AI Politician - DETAILED TRACE MODE")
    print("=" * 50)
    print("This mode shows the complete internal workflow with detailed information:")
    print("  1. Context Agent - Extracts topics and retrieves knowledge")
    print("  2. Sentiment Agent - Analyzes tone and emotions")
    print("  3. Response Agent - Generates the final response")
    print()
    print("Each component will show:")
    print("  • Input data received")
    print("  • Processing details")
    print("  • Output produced")
    print()
    print("The politician's complete response will be shown at the end of each interaction.")
    print("=" * 50, "\n")
    
    # Execute the command
    os.system(command)

if __name__ == "__main__":
    main() 