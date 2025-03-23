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
    
    # API component
    api_parser = subparsers.add_parser("api", help="Launch the API server")
    
    # Parse only the component argument first
    args, remaining = parser.parse_known_args()
    
    # Handle the specified component
    if args.component == "cli":
        # Import the CLI module
        from src.models.langgraph.cli import main
        
        # Run the CLI with the remaining arguments
        sys.argv = [sys.argv[0]] + remaining
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