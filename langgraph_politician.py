#!/usr/bin/env python3
"""
Main launcher script for the AI Politician LangGraph system.
This script provides a command-line interface to launch different components of the system.
"""
import sys
import argparse
from pathlib import Path

# Add the project root to the Python path
root_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(root_dir))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Politician LangGraph System")
    
    # Subparsers for different components
    subparsers = parser.add_subparsers(dest="component", help="Component to launch")
    
    # CLI component
    cli_parser = subparsers.add_parser("cli", help="Launch the command-line interface")
    cli_parser.add_argument("args", nargs="*", help="Arguments to pass to the CLI")
    
    # API component
    api_parser = subparsers.add_parser("api", help="Launch the API server")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle the specified component
    if args.component == "cli":
        # Import and run the CLI with the provided arguments
        from src.models.langgraph.cli import main
        
        # Replace sys.argv with the CLI arguments
        sys.argv = [sys.argv[0]] + args.args
        main()
    elif args.component == "api":
        # Import and run the API server
        from src.models.langgraph.api import main
        main()
    else:
        # Display usage information
        print("Usage: langgraph_politician.py {cli|api} [args...]")
        print("\nExamples:")
        print("  langgraph_politician.py cli chat --identity biden --debug")
        print("  langgraph_politician.py cli visualize")
        print("  langgraph_politician.py api")
        
        # If no component was specified, display help
        parser.print_help() 