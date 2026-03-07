#!/usr/bin/env python3
"""
Helper script to run the AI Politician Debate System.
This script provides a simple command-line interface to start debates.
"""
import sys
import logging
import argparse
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(root_dir))

# Set logging level to reduce verbosity
logging.basicConfig(level=logging.ERROR)

from scripts._subprocess_utils import run_script

def main():
    """Run the AI Politician Debate System."""
    parser = argparse.ArgumentParser(description="Run AI Politician Debate")
    parser.add_argument("--topic", type=str, default="General Political Discussion", 
                      help="Debate topic")
    parser.add_argument("--format", type=str, default="head_to_head",
                      choices=["town_hall", "head_to_head", "panel"],
                      help="Debate format")
    parser.add_argument("--participants", type=str, default="biden,trump",
                      help="Comma-separated list of participants")
    parser.add_argument("--time-per-turn", type=int, default=60,
                      help="Time in seconds per turn")
    parser.add_argument("--allow-interruptions", action="store_true",
                      help="Allow interruptions")
    parser.add_argument("--no-fact-check", action="store_true",
                      help="Disable fact checking")
    parser.add_argument("--moderator-control", type=str, default="moderate",
                      choices=["strict", "moderate", "minimal"],
                      help="Level of moderator control")
    parser.add_argument("--no-rag", action="store_true",
                      help="Disable RAG knowledge retrieval")
    parser.add_argument("--trace", action="store_true",
                      help="Show trace information")
    
    args = parser.parse_args()
    
    # Build the command to run the debate
    script_path = root_dir / "langgraph_politician.py"
    command = [
        sys.executable,
        str(script_path),
        "debate",
        "run",
        "--topic",
        args.topic,
        "--participants",
        args.participants,
        "--format",
        args.format,
    ]
    
    # Add optional arguments
    if args.time_per_turn != 60:
        command.extend(["--time-per-turn", str(args.time_per_turn)])
    if args.allow_interruptions:
        command.append("--allow-interruptions")
    if args.no_fact_check:
        command.append("--no-fact-check")
    if args.moderator_control != "moderate":
        command.extend(["--moderator-control", args.moderator_control])
    if args.no_rag:
        command.append("--no-rag")
    if args.trace:
        command.append("--trace")
    
    print("\n🎤 AI Politician Debate System 🎤")
    print("=" * 50)
    print(f"Starting debate on topic: {args.topic}")
    print(f"Format: {args.format}")
    print(f"Participants: {args.participants}")
    print("=" * 50, "\n")
    
    # Execute the command
    exit_code = run_script(command)
    if exit_code:
        raise SystemExit(exit_code)

if __name__ == "__main__":
    main() 
