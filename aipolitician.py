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
import sys
import os
import argparse
from pathlib import Path

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
    
    # Parse arguments
    args = parser.parse_args()
    
    # Get script directory
    script_dir = Path(__file__).parent / "scripts"
    
    # Handle the specified mode
    if args.mode == "chat":
        script = str(script_dir / "chat" / "chat_politician.py")
        command = f"python {script} {args.identity}"
        if args.no_rag:
            command += " --no-rag"
        os.system(command)
    
    elif args.mode == "debug":
        script = str(script_dir / "chat" / "debug_politician.py")
        command = f"python {script} {args.identity}"
        if args.no_rag:
            command += " --no-rag"
        os.system(command)
    
    elif args.mode == "trace":
        script = str(script_dir / "chat" / "trace_politician.py")
        command = f"python {script} {args.identity}"
        if args.no_rag:
            command += " --no-rag"
        os.system(command)
    
    elif args.mode == "debate":
        script = str(script_dir / "debate" / "debate_politician.py")
        command = f"python {script} --topic \"{args.topic}\" --format {args.format}"
        if args.no_rag:
            command += " --no-rag"
        os.system(command)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 