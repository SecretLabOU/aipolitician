#!/usr/bin/env python3
"""
Political Chat Interface

This script provides a unified interface for chatting with political personas
using the LangGraph integration for more advanced conversation capabilities.
"""

import os
import sys
import argparse
import asyncio
from pathlib import Path

# Add the lang-graph/src directory to the Python path
LANG_GRAPH_DIR = Path(__file__).parent / "lang-graph" / "src"
sys.path.append(str(LANG_GRAPH_DIR))

# Define persona mappings
PERSONA_MAP = {
    "trump": "donald_trump",
    "biden": "joe_biden"
}

PERSONA_DISPLAY_NAMES = {
    "donald_trump": "Donald Trump",
    "joe_biden": "Joe Biden"
}

# Try to import from the lang-graph module
try:
    from political_agent_graph.config import select_persona, get_active_persona
    from political_agent_graph.graph import run_conversation
    from political_agent_graph.local_models import setup_models, get_model
    LANG_GRAPH_AVAILABLE = True
except ImportError as e:
    print(f"Error importing lang-graph module: {e}")
    print("Running without lang-graph integration.")
    LANG_GRAPH_AVAILABLE = False

# RAG system availability check
try:
    from embeddings import query_rag_db
    RAG_AVAILABLE = True
except ImportError:
    print("RAG database system not available. Running without RAG.")
    RAG_AVAILABLE = False

class PoliticalChat:
    """Main class for the political chat interface."""
    
    def __init__(self, persona=None, use_rag=False, max_length=512):
        """Initialize the chat interface."""
        self.use_rag = use_rag and RAG_AVAILABLE
        self.max_length = max_length
        
        if LANG_GRAPH_AVAILABLE:
            setup_models()
            
            # Set the persona if provided, otherwise it will use the default
            if persona:
                select_persona(persona)
            
            self.active_persona = get_active_persona()
        else:
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
        """Generate a response based on the selected persona."""
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
        print("Type 'toggle rag' to enable/disable RAG mode.")
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
    parser = argparse.ArgumentParser(description="Chat with political personas (Advanced LangGraph version)")
    
    # Add command line arguments
    parser.add_argument("--persona", choices=["trump", "biden"], 
                      help="Select persona to chat with (trump or biden)")
    parser.add_argument("--rag", action="store_true", 
                      help="Enable RAG mode for enhanced responses")
    parser.add_argument("--max-length", type=int, default=512, 
                      help="Maximum length of generated responses")
    
    args = parser.parse_args()
    selected_persona = None
    
    # Display menu for persona selection if not specified via command line
    if args.persona:
        # Convert command line arg to internal persona ID
        selected_persona = PERSONA_MAP.get(args.persona.lower())
    else:
        print("\n==== Political Chat Interface (LangGraph) ====")
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
