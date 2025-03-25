#!/usr/bin/env python3
"""
Political Chat Interface

This script provides a unified interface for chatting with different political personas.
It integrates with the lang-graph module for enhanced conversational capabilities.
"""

import os
import sys
import argparse
import asyncio
import readline
import traceback
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

# Add the lang-graph/src directory to the Python path
LANG_GRAPH_DIR = Path(__file__).parent / "lang-graph" / "src"
sys.path.append(str(LANG_GRAPH_DIR))

# Try to import from the lang-graph module
try:
    from political_agent_graph.config import (
        PERSONA_MODEL_MAP,
        select_persona,
        get_active_persona
    )
    from political_agent_graph.graph import run_conversation
    from political_agent_graph.local_models import setup_models, get_model
    from political_agent_graph.state import ConversationState, get_initial_state
    LANG_GRAPH_AVAILABLE = True
except ImportError as e:
    print(f"Error importing lang-graph module: {e}")
    print("Running without lang-graph integration.")
    LANG_GRAPH_AVAILABLE = False

# RAG system availability check
try:
    from embeddings import get_rag_db, query_rag_db
    RAG_AVAILABLE = True
except ImportError:
    print("RAG database system not available. Running without RAG.")
    RAG_AVAILABLE = False

# Define persona display names mapping
PERSONA_DISPLAY_NAMES = {
    "donald_trump": "Donald Trump",
    "joe_biden": "Joe Biden"
}

# Fallback persona map if lang-graph is not available
FALLBACK_PERSONA_MAP = {
    "trump": "donald_trump",
    "biden": "joe_biden"
}

class PoliticalChat:
    """Main class for the political chat interface."""
    
    def __init__(self, persona=None, use_rag=False, max_length=512):
        """Initialize the chat interface.
        
        Args:
            persona: The persona to use (donald_trump or joe_biden)
            use_rag: Whether to use RAG for enhanced responses
            max_length: Maximum length of generated responses
        """
        self.use_rag = use_rag and RAG_AVAILABLE
        self.max_length = max_length
        
        if LANG_GRAPH_AVAILABLE:
            setup_models()
            initialize_personas()
            
            # Set the persona if provided, otherwise it will use the default
            if persona:
                set_active_persona(persona)
            
            self.active_persona = get_active_persona()
        else:
            print("Lang-graph integration not available. Using fallback model loading.")
            if persona:
                self.active_persona = persona
            else:
                self.active_persona = "donald_trump"  # Default
    
    def toggle_rag(self):
        """Toggle RAG mode on/off."""
        if not RAG_AVAILABLE:
            print("RAG is not available in this environment.")
            return False
        
        self.use_rag = not self.use_rag
        rag_status = "ON" if self.use_rag else "OFF"
        print(f"RAG mode is now {rag_status}")
        return self.use_rag
    
    def get_rag_status(self):
        """Get current RAG status."""
        if not RAG_AVAILABLE:
            return "unavailable"
        return "enabled" if self.use_rag else "disabled"
    
    async def generate_response(self, user_input):
        """Generate a response based on the selected persona.
        
        Args:
            user_input: The user's message
            
        Returns:
            The generated response
        """
        if not user_input.strip():
            return "Please enter a question or statement."
        
        # Use RAG if enabled
        context = ""
        if self.use_rag:
            try:
                context = query_rag_db(user_input)
                print("\nUsing RAG context for enhanced response...")
            except Exception as e:
                print(f"Error using RAG: {e}")
        
        # Generate response using lang-graph if available
        if LANG_GRAPH_AVAILABLE:
            try:
                # Prepend RAG context if available
                full_input = f"{context}\n\nUser question: {user_input}" if context else user_input
                response = await run_conversation(full_input)
                return response
            except Exception as e:
                print(f"Error using lang-graph: {e}")
                print("Falling back to basic generation...")
        
        # Fallback to direct model usage if lang-graph fails or is not available
        try:
            model_name = "trump" if self.active_persona == "donald_trump" else "biden"
            model = get_model(model_name)
            
            full_input = f"{context}\n\nUser question: {user_input}" if context else user_input
            
            # We're using the _call method directly since we're not using the LangChain interface
            response = model._call(full_input, max_length=self.max_length)
            return response
        except Exception as e:
            return f"Error generating response: {e}"
    
    async def run_chat_session(self):
        """Run an interactive chat session with the selected persona."""
        persona_display_name = PERSONA_DISPLAY_NAMES.get(self.active_persona, self.active_persona)
        
        print(f"\n{'='*60}")
        print(f"Starting chat with {persona_display_name}")
        print(f"RAG mode: {self.get_rag_status().upper()}")
        print("Type 'exit', 'quit', or Ctrl+C to end the conversation.")
        print(f"{'='*60}\n")
        
        while True:
            try:
                user_input = input("\nYou: ")
                
                # Check for exit commands
                if user_input.lower() in ["exit", "quit"]:
                    break
                
                # Check for RAG toggle command
                if user_input.lower() == "toggle rag":
                    self.toggle_rag()
                    print(f"RAG mode is now {'ON' if self.use_rag else 'OFF'}")
                    continue
                
                # Generate and display response
                response = await self.generate_response(user_input)
                print(f"\n{persona_display_name}: {response}")
                
            except KeyboardInterrupt:
                print("\nExiting chat session...")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue

async def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Chat with political personas")
    
    # Add command line arguments
    parser.add_argument("--persona", choices=["trump", "biden"], 
                      help="Select persona to chat with (trump or biden)")
    parser.add_argument("--rag", action="store_true", 
                      help="Enable RAG mode for enhanced responses")
    parser.add_argument("--max-length", type=int, default=512, 
                      help="Maximum length of generated responses")
    
    args = parser.parse_args()
    # Map command line persona names to internal persona IDs
    persona_map = PERSONA_MODEL_MAP if LANG_GRAPH_AVAILABLE else FALLBACK_PERSONA_MAP
    selected_persona = None
    
    # Display menu for persona selection if not specified via command line
    if args.persona:
        # Convert command line arg to internal persona ID
        selected_persona = persona_map.get(args.persona.lower(), None)
    else:
        print("\n==== Political Chat Interface ====")
        print("1. Chat with Donald Trump")
        print("2. Chat with Joe Biden")
        print("q. Quit")
        
        choice = input("\nSelect an option: ")
        
        if choice == "1":
            selected_persona = "donald_trump"
        elif choice == "2":
            selected_persona = "joe_biden"
        elif choice.lower() in ["q", "quit", "exit"]:
            print("Exiting.")
            return
        else:
            print("Invalid choice. Defaulting to Donald Trump.")
            selected_persona = "donald_trump"
    
    # Initialize and run the chat interface
    chat = PoliticalChat(
        persona=selected_persona,
        use_rag=args.rag,
        max_length=args.max_length
    )
    
    await chat.run_chat_session()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting.")
        sys.exit(0)

# Define our own versions of missing functions
# Implement our own versions of missing functions
_rag_mode = False  # Default RAG mode to False
# Function no longer needed as we directly use PERSONA_MODEL_MAP.keys()
# def get_persona_names():
#     """Get the list of available personas from PERSONA_MODEL_MAP."""
#     return list(PERSONA_MODEL_MAP.keys())

def toggle_rag_mode(enable=None):
    """Toggle or set the RAG mode."""
    global _rag_mode
    if enable is not None:
        _rag_mode = enable
    else:
        _rag_mode = not _rag_mode
    return _rag_mode

def get_rag_mode():
    """Get the current RAG mode state."""
    return _rag_mode

def print_header(title: str) -> None:
    """Print a header with the title centered."""
    width = 80
    print("\n" + "=" * width)
    print(f"{title.center(width)}")
    print("=" * width)


def print_menu(options: List[str], header: str = None) -> None:
    """Print a menu with numbered options."""
    if header:
        print_header(header)
    
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    print(f"q. Quit")


def get_menu_choice(options: List[str]) -> Union[int, str]:
    """Get a user's menu choice and validate it."""
    while True:
        choice = input("\nEnter your choice: ").strip().lower()
        if choice == 'q':
            return 'q'
        
        try:
            choice_num = int(choice)
            if 1 <= choice_num <= len(options):
                return choice_num
            print(f"Invalid choice. Please enter a number between 1 and {len(options)}.")
        except ValueError:
            print("Invalid input. Please enter a number or 'q' to quit.")


def select_persona_menu() -> None:
    """Display a menu for selecting a political persona."""
    # Use the actual personas from PERSONA_MODEL_MAP
    personas = list(PERSONA_MODEL_MAP.keys())
    
    if not personas:
        print("No personas found. Please check your installation.")
        return
    
    # Format persona names for display
    display_names = [p.replace("_", " ").title() for p in personas]
    
    print_menu(display_names, "Select a Political Persona")
    choice = get_menu_choice(display_names)
    
    if choice == 'q':
        return
    
    try:
        # Use the original key from PERSONA_MODEL_MAP
        persona_name = personas[choice - 1]
        select_persona(persona_name)
        display_name = persona_name.replace("_", " ").title()
        print(f"\nPersona switched to: {display_name}")
    except Exception as e:
        print(f"Error selecting persona: {str(e)}")


def toggle_rag_menu() -> None:
    """Display a menu for toggling RAG mode."""
    current_mode = "ON" if get_rag_mode() else "OFF"
    print_header(f"Toggle RAG Mode (Currently: {current_mode})")
    
    options = ["Enable RAG", "Disable RAG"]
    choice = get_menu_choice(options)
    
    if choice == 'q':
        return
    
    try:
        if choice == 1:
            toggle_rag_mode(True)
            print("\nRAG mode enabled.")
        else:
            toggle_rag_mode(False)
            print("\nRAG mode disabled.")
    except Exception as e:
        print(f"Error toggling RAG mode: {str(e)}")


def display_help() -> None:
    """Display comprehensive help information."""
    print_header("Political Chat Help")
    
    print("""
OVERVIEW:
---------
This application provides an interactive chat interface with various political personas.
You can have conversations with AI models trained to respond in the style of different
political figures.

AVAILABLE PERSONAS:
------------------
- Joe Biden: Current US President (Democrat)
- Donald Trump: Former US President (Republican)
- Additional personas may be available based on your installation

FEATURES:
---------
- Switch between different political personas during your chat session
- Toggle RAG (Retrieval Augmented Generation) mode on/off
- Save your conversation history
- Configure model parameters

COMMANDS:
---------
During chat, you can use the following special commands:
- /menu   : Return to the main menu
- /quit   : Exit the application
- /help   : Display this help information
- /clear  : Clear the current conversation
- /persona: Switch to a different persona
- /rag    : Toggle RAG mode on/off

COMMAND LINE ARGUMENTS:
----------------------
--rag         : Start with RAG mode enabled (default: disabled)
--max-length N: Set maximum response length (default: 1024)
--max-tokens N: Set maximum tokens per response (default: 1024)
--temperature X: Set response randomness (0.0-1.0, default: 0.7)
--persona NAME: Start with specified persona
--help        : Display command-line help
    """)
    
    input("\nPress Enter to continue...")


def main_menu() -> bool:
    """Display the main menu and handle user selections. Returns False to exit."""
    active_persona_id = get_active_persona()
    # Format the persona name for display
    persona_name = active_persona_id.replace("_", " ").title() if active_persona_id else "None"
    rag_status = "ON" if get_rag_mode() else "OFF"
    
    options = [
        f"Chat with {persona_name}",
        "Select different persona",
        f"Toggle RAG mode (currently: {rag_status})",
        "View help",
    ]
    
    print_menu(options, "Political Chat Interface")
    choice = get_menu_choice(options)
    
    if choice == 'q':
        return False
    
    if choice == 1:  # Chat with current persona
        return True
    elif choice == 2:  # Select different persona
        select_persona_menu()
    elif choice == 3:  # Toggle RAG mode
        toggle_rag_menu()
    elif choice == 4:  # View help
        display_help()
    
    return True


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive political chat with AI personas."
    )
    
    parser.add_argument(
        "--rag",
        action="store_true",
        help="Enable RAG mode (Retrieval Augmented Generation)",
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum length of generated responses",
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens in generated responses",
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for response generation (0.0-1.0)",
    )
    
    parser.add_argument(
        "--persona",
        type=str,
        help="Start with the specified persona (e.g., 'biden', 'trump')",
    )
    
    return parser.parse_args()


async def chat_loop(state: ConversationState) -> None:
    """Run the main chat loop."""
    while True:
        try:
            # Get current persona info for prompt
            active_persona_id = get_active_persona()
            persona_name = active_persona_id.replace("_", " ").title() if active_persona_id else "Unknown"
            
            # Get current RAG status
            rag_status = "ON" if get_rag_mode() else "OFF"
            
            # Display prompt with persona information
            user_input = input(f"\n[Chatting with {persona_name} | RAG: {rag_status}] You: ")
            
            # Handle special commands
            if user_input.lower() in ['/quit', '/exit']:
                break
            elif user_input.lower() == '/menu':
                if not main_menu():
                    break
                continue
            elif user_input.lower() == '/help':
                display_help()
                continue
            elif user_input.lower() == '/clear':
                state = get_initial_state("")
                print("Conversation cleared.")
                continue
            elif user_input.lower() == '/persona':
                select_persona_menu()
                continue
            elif user_input.lower() == '/rag':
                toggle_rag_menu()
                continue
            
            # Skip empty inputs
            if not user_input.strip():
                continue
            
            # Process the user's input through the political agent graph
            state.user_input = user_input
            response = await run_conversation(user_input)
            state.final_response = response
            state.agent_response = response
            
            # Display the agent's response
            print(f"\n{persona_name}: {state.agent_response}")
            
        except KeyboardInterrupt:
            print("\nExiting chat...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Try again or type /menu to return to the main menu.")
            print(traceback.format_exc())


async def main() -> None:
    """Main entry point for the political chat application."""
    args = parse_arguments()
    
    try:
        # Initialize the system
        print_header("Political Chat - Initializing")
        print("Loading language models and knowledge base...")
        
        # Set RAG mode based on command line argument
        if args.rag:
            toggle_rag_mode(True)
            print("RAG mode enabled.")
        
        # The political agent graph is now initialized automatically when imported
        # No need to explicitly initialize it
        
        # Set initial persona if specified
        if args.persona:
            try:
                # Ensure persona name matches one of the keys in PERSONA_MODEL_MAP
                persona_arg = args.persona.lower()
                valid_personas = PERSONA_MODEL_MAP.keys()
                
                # Find the closest matching persona
                matching_persona = None
                for persona in valid_personas:
                    if persona.lower() == persona_arg or persona.lower().replace("_", "") == persona_arg.replace("_", ""):
                        matching_persona = persona
                        break
                
                if matching_persona:
                    select_persona(matching_persona)
                    print(f"Using persona: {matching_persona}")
                else:
                    print(f"Persona '{args.persona}' not found. Valid personas: {', '.join(valid_personas)}")
                    print("Using default persona instead.")
            except Exception as e:
                print(f"Error setting initial persona: {str(e)}")
                print("Using default persona instead.")
        # Create initial state
        state = get_initial_state("")
        # Show main menu first unless a persona was explicitly specified
        if not args.persona:
            print("\nWelcome to the Political Chat Interface!")
            if not main_menu():
                print("Goodbye!")
                return
        
        # Start the chat loop
        await chat_loop(state)
        
    except Exception as e:
        print(f"Error initializing the application: {str(e)}")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting.")
        sys.exit(0)
