#!/usr/bin/env python3
"""
AI Politician - Main Launcher
=============================

This is the unified launcher script for the AI Politician system.
It provides easy access to all available modes in one place.

Usage:
  python aipolitician.py chat biden     # Clean chat mode
  python aipolitician.py debug biden    # Debug mode with analysis info
  python aipolitician.py trace biden    # Trace mode with detailed output
  python aipolitician.py debate         # Run a debate between politicians
  
  Add --no-rag to any command to disable the knowledge database.
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path


def run_script(script_path: Path, script_args: list[str]) -> int:
    """Run a python script and return its exit code."""
    command = ["python", str(script_path), *script_args]
    completed = subprocess.run(command, check=False)
    return completed.returncode

def main():
    """Main entry point for the launcher."""
    parser = argparse.ArgumentParser(
        description="AI Politician - Unified Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python aipolitician.py chat biden     # Clean chat experience
  python aipolitician.py debug biden    # Show debugging info
  python aipolitician.py trace biden    # Show detailed workflow tracing
  python aipolitician.py debate         # Run a debate between politicians
        """
    )
    
    # Add subparsers for different modes
    subparsers = parser.add_subparsers(dest="mode", help="Operating mode")
    
    # Chat mode
    chat_parser = subparsers.add_parser("chat", help="Clean chat experience")
    chat_parser.add_argument("identity", choices=["biden", "trump"], help="Politician identity")
    chat_parser.add_argument("--no-rag", action="store_true", help="Disable RAG database")
    
    # Debug mode
    debug_parser = subparsers.add_parser("debug", help="Debug mode with analysis info")
    debug_parser.add_argument("identity", choices=["biden", "trump"], help="Politician identity")
    debug_parser.add_argument("--no-rag", action="store_true", help="Disable RAG database")
    
    # Trace mode
    trace_parser = subparsers.add_parser("trace", help="Trace mode with detailed output")
    trace_parser.add_argument("identity", choices=["biden", "trump"], help="Politician identity")
    trace_parser.add_argument("--no-rag", action="store_true", help="Disable RAG database")
    
    # Debate mode
    debate_parser = subparsers.add_parser("debate", help="Run a debate between politicians")
    debate_parser.add_argument("--topic", type=str, default="General Political Discussion", 
                            help="Debate topic")
    debate_parser.add_argument("--format", type=str, default="head_to_head",
                            choices=["town_hall", "head_to_head", "panel"],
                            help="Debate format")
    debate_parser.add_argument("--no-rag", action="store_true", help="Disable RAG database")
    debate_parser.add_argument("--no-fact-check", action="store_true", help="Disable fact checking")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Get script directory
    script_dir = Path(__file__).parent / "scripts"
    
    exit_code = 0

    # Handle the specified mode
    if args.mode == "chat":
        script = script_dir / "chat" / "chat_politician.py"
        script_args = [args.identity]
        if args.no_rag:
            script_args.append("--no-rag")
        exit_code = run_script(script, script_args)

    elif args.mode == "debug":
        script = script_dir / "chat" / "debug_politician.py"
        script_args = [args.identity]
        if args.no_rag:
            script_args.append("--no-rag")
        exit_code = run_script(script, script_args)

    elif args.mode == "trace":
        script = script_dir / "chat" / "trace_politician.py"
        script_args = [args.identity]
        if args.no_rag:
            script_args.append("--no-rag")
        exit_code = run_script(script, script_args)

    elif args.mode == "debate":
        script = script_dir / "debate" / "debate_politician.py"
        script_args = ["--topic", args.topic, "--format", args.format]
        if args.no_rag:
            script_args.append("--no-rag")
        if args.no_fact_check:
            script_args.append("--no-fact-check")
        exit_code = run_script(script, script_args)

    else:
        parser.print_help()
        return

    if exit_code:
        mode = args.mode or "command"
        print(f"Error: {mode} exited with status {exit_code}", file=sys.stderr)
        raise SystemExit(exit_code)

if __name__ == "__main__":
    main() 
