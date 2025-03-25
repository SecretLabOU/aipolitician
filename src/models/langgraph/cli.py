#!/usr/bin/env python3
"""
Command-line interface for the AI Politician LangGraph system.
"""
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add the project root to the Python path
root_dir = Path(__file__).parent.parent.parent.parent.absolute()
sys.path.insert(0, str(root_dir))

from src.models.langgraph.config import PoliticianIdentity
from src.models.langgraph.workflow import process_user_input, PoliticianInput
from src.models.langgraph.utils.visualization import visualize_graph

# Configure logging to reduce verbose output
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("peft").setLevel(logging.ERROR)
logging.getLogger("safetensors").setLevel(logging.ERROR)

def format_sentiment_analysis(analysis: Dict[str, Any]) -> str:
    """Format the sentiment analysis for display."""
    result = "\nSentiment Analysis:\n"
    result += f"- Score: {analysis.get('sentiment_score', 0.0):.2f} (-1.0 to 1.0)\n"
    result += f"- Category: {analysis.get('sentiment_category', 'unknown')}\n"
    result += f"- Biased Question: {'Yes' if analysis.get('is_biased', False) else 'No'}\n"
    result += f"- Contains Personal Attack: {'Yes' if analysis.get('contains_personal_attack', False) else 'No'}\n"
    result += f"- 'Gotcha' Question: {'Yes' if analysis.get('is_gotcha_question', False) else 'No'}\n"
    return result

def chat_loop(politician_identity: str, use_rag: bool = True, debug: bool = False, trace: bool = False):
    """Interactive chat loop with the AI Politician."""
    # Print welcome message
    if politician_identity == PoliticianIdentity.BIDEN:
        print("\n🇺🇸 Biden AI Chat 🇺🇸")
    elif politician_identity == PoliticianIdentity.TRUMP:
        print("\n🇺🇸 Trump AI Chat 🇺🇸")
    else:
        print("\n🇺🇸 AI Politician Chat 🇺🇸")
    
    print("===================")
    print(f"This is an AI simulation of {politician_identity.title()}'s speaking style and policy positions.")
    print("Type 'quit' or press Ctrl+C to end the conversation.")
    
    # Print example prompts
    print("\nExample prompts:")
    if politician_identity == PoliticianIdentity.BIDEN:
        print("1. What's your vision for America's future?")
        print("2. How would you handle the situation at the southern border?")
        print("3. Tell me about your infrastructure plan")
        print("4. What do you think about Donald Trump?")
        print("5. How are you addressing climate change?")
    elif politician_identity == PoliticianIdentity.TRUMP:
        print("1. What's your plan for the economy?")
        print("2. How would you handle the situation in Ukraine?")
        print("3. Tell me about your greatest achievements as president")
        print("4. What do you think about Joe Biden?")
        print("5. How would you make America great again?")
    
    # Start the chat loop
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ")
            if user_input.lower() in ["quit", "exit"]:
                break
            
            # Process the input
            input_data = PoliticianInput(
                user_input=user_input,
                politician_identity=politician_identity,
                use_rag=use_rag,
                trace=trace
            )
            
            # For clean chat mode (not trace or debug), show a loading indicator
            if not trace and not debug:
                print(f"\n{politician_identity.title()}: ", end="", flush=True)
            
            # Process through the graph
            result = process_user_input(input_data)
            
            # Print the response based on mode
            if trace:
                # Trace mode - display with clear separation and formatting
                print(f"\n{politician_identity.title()}'s Response:")
                print("---------------------")
                print(result.response)
                print("---------------------")
            elif debug:
                # Debug mode - show response plus debug info
                print(f"\n{politician_identity.title()}: {result.response}")
                print("\nDebug Information:")
                print("-----------------")
                print(format_sentiment_analysis(result.sentiment_analysis))
                print(f"Relevant Knowledge Found: {'Yes' if result.has_knowledge else 'No'}")
                print(f"Deflection Used: {'Yes' if result.should_deflect else 'No'}")
                print("-----------------")
            else:
                # Clean chat mode - just show the response like a normal conversation
                print(result.response)
            
        except KeyboardInterrupt:
            print("\nExiting chat...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="AI Politician LangGraph System")
    
    # Subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start an interactive chat with the AI Politician")
    chat_parser.add_argument("--identity", type=str, choices=["biden", "trump"], required=True,
                         help="Identity of the politician to impersonate")
    chat_parser.add_argument("--no-rag", action="store_true", 
                         help="Disable Retrieval-Augmented Generation")
    chat_parser.add_argument("--debug", action="store_true",
                         help="Enable debug mode with additional output")
    chat_parser.add_argument("--trace", action="store_true",
                         help="Enable tracing to show the workflow execution path")
    
    # Process input command
    process_parser = subparsers.add_parser("process", help="Process a single input and return JSON output")
    process_parser.add_argument("--identity", type=str, choices=["biden", "trump"], required=True,
                             help="Identity of the politician to impersonate")
    process_parser.add_argument("--no-rag", action="store_true",
                             help="Disable Retrieval-Augmented Generation")
    process_parser.add_argument("--input", type=str, required=True,
                             help="Input prompt to process")
    process_parser.add_argument("--trace", action="store_true",
                             help="Enable tracing to show the workflow execution path")
    
    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Generate a visualization of the workflow graph")
    viz_parser.add_argument("--output", type=str, 
                         help="Path to save the visualization HTML file")
    viz_parser.add_argument("--no-open", action="store_true",
                         help="Don't automatically open the visualization in a browser")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle the specified command
    if args.command == "chat":
        chat_loop(
            politician_identity=args.identity,
            use_rag=not args.no_rag,
            debug=args.debug,
            trace=args.trace
        )
    elif args.command == "process":
        # Process a single input and return JSON output
        input_data = PoliticianInput(
            user_input=args.input,
            politician_identity=args.identity,
            use_rag=not args.no_rag,
            trace=args.trace
        )
        
        result = process_user_input(input_data)
        print(json.dumps(result.dict(), indent=2))
    elif args.command == "visualize":
        # Generate a visualization of the graph
        output_path = visualize_graph(
            output_path=args.output,
            auto_open=not args.no_open
        )
        print(f"Graph visualization saved to: {output_path}")
    else:
        parser.print_help()

def run_cli():
    """Run the CLI directly (for use in imported contexts)."""
    main()

if __name__ == "__main__":
    main() 