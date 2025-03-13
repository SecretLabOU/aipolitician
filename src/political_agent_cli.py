#!/usr/bin/env python3
"""Political Persona Conversation System.

Talk to AI politicians with a single command.
"""

import os
import asyncio
import argparse

from political_agent_graph import (
    run_conversation, 
    select_persona,
    persona_manager
)

def display_personas():
    """Display all available personas."""
    print("\nAvailable Politicians:")
    for persona_id, persona in persona_manager.personas.items():
        print(f"  {persona['name']} - use: {persona_id}")
    print()

async def chat_with(persona_id: str):
    """Start a conversation with a politician."""
    try:
        select_persona(persona_id)
        persona = persona_manager.get_active_persona()
        
        print(f"\nTalking with: {persona['name']} ({persona['party']})")
        print("Ask any question or type 'exit' to quit\n")
        
        while True:
            user_input = input("> ")
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\nGoodbye!")
                break
            
            print("Thinking...", end="\r")
            response = await run_conversation(user_input)
            print(" " * 20, end="\r")  # Clear "Thinking..."
            print(f"\n{persona['name']}: {response}\n")
    
    except KeyboardInterrupt:
        print("\nConversation ended.")
    except Exception as e:
        print(f"\nError: {e}")

async def quick_demo():
    """Run a quick demonstration with several politicians."""
    print("\n=== QUICK DEMO ===\n")
    
    politicians = [
        ("donald_trump", "What's your view on immigration?"),
        ("joe_biden", "What do you think about healthcare?")
    ]
    
    for persona_id, question in politicians:
        select_persona(persona_id)
        persona = persona_manager.get_active_persona()
        
        print(f"{persona['name']}:")
        print(f"Q: {question}")
        print("Thinking...", end="\r")
        
        response = await run_conversation(question)
        print(" " * 20, end="\r")  # Clear "Thinking..."
        print(f"A: {response}\n")
        print("-" * 40)
    
    print("\nDemo complete!\n")

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Talk to AI Politicians")
    parser.add_argument("politician", nargs="?", help="Politician to talk to (e.g., donald_trump)")
    parser.add_argument("--demo", action="store_true", help="See quick examples")
    parser.add_argument("--list", action="store_true", help="List available politicians")
    args = parser.parse_args()
    
    if args.list:
        display_personas()
        return
        
    if args.demo:
        await quick_demo()
        return
    
    # If politician specified, talk to them directly
    if args.politician and args.politician in persona_manager.personas:
        await chat_with(args.politician)
    # Otherwise show options and ask
    else:
        display_personas()
        persona_id = input("Who would you like to talk to? ")
        if persona_id in persona_manager.personas:
            await chat_with(persona_id)
        else:
            print(f"Unknown politician: {persona_id}")
            display_personas()

if __name__ == "__main__":
    asyncio.run(main())