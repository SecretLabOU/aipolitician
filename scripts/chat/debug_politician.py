#!/usr/bin/env python3
"""
Helper script to run the AI Politician system in debug mode.
This script provides a chat experience with additional debugging information.
"""
import sys
import logging
import subprocess
import argparse
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(root_dir))

# Set logging level to reduce verbosity
logging.basicConfig(level=logging.ERROR)


def run_script(command: list[str]) -> int:
    """Run a command and return its exit status."""
    try:
        completed = subprocess.run(command, check=False)
    except KeyboardInterrupt:
        return 130
    except OSError as exc:
        print(f"Error: failed to execute command: {exc}", file=sys.stderr)
        return 127
    if completed.returncode < 0:
        return 128 + abs(completed.returncode)
    return completed.returncode

def main():
    """Run the AI Politician system in debug mode."""
    parser = argparse.ArgumentParser(description="Chat with AI Politician (Debug Mode)")
    parser.add_argument("identity", choices=["biden", "trump"], help="Politician identity")
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG database")
    
    args = parser.parse_args()
    
    # Build the command to run the system in debug mode
    script_path = root_dir / "langgraph_politician.py"
    command = [sys.executable, str(script_path), "chat", "--identity", args.identity, "--debug"]
    if args.no_rag:
        command.append("--no-rag")
    
    print("\n🔍 AI Politician - DEBUG MODE")
    print("=" * 50)
    print("Starting chat experience with additional debugging information.")
    print("You'll see sentiment analysis and other debugging details.")
    print("=" * 50, "\n")
    
    # Execute the command
    completed = run_script(command)
    if completed:
        raise SystemExit(completed)

if __name__ == "__main__":
    main() 
