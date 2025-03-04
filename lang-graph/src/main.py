#!/usr/bin/env python3
"""Political Persona Conversation System.

Talk to AI politicians with a single command.
"""

import os
import asyncio
import argparse

from political_agent_graph import (
    run_conversation, 
    get_available_personas, 
    select_persona,
    persona_manager
)

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_welcome():
    """Print welcome message."""
    clear_screen()
    print("=" * 60)
    print(f"{'POLITICAL AI':^60}")
    print("=" * 60)
    print("\nTalk to AI politicians and get their views on any topic")
    print("=" * 60)
    print()

def display_personas():
    """Display all available personas."""
    print("\nChoose a politician:")
    for persona_id, persona in persona_manager.personas.items():
        print(f"  {persona['name']} - just type: {persona_id}")
    print()

async def chat_with(persona_id: str):
    """Start a conversation with a politician."""
    try:
        select_persona(persona_id)
        persona = persona_manager.get_active_persona()
        
        print(f"\nNow talking with: {persona['name']} ({persona['party']})")
        print("\nAsk any question or type 'exit' to quit\n")
        
        while True:
            user_input = input("> ")
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\nGoodbye!")
                break
            
            print("\nThinking...")
            response = await run_conversation(user_input)
            print(f"\n{persona['name']}: {response}\n")
    
    except KeyboardInterrupt:
        print("\nConversation ended.")
    except Exception as e:
        print(f"\nError: {e}")

async def quick_demo():
    """Run a quick demonstration with several politicians."""
    print("\n=== QUICK DEMO ===\n")
    
    politicians = [
        ("bernie_sanders", "What do you think about healthcare?"),
        ("donald_trump", "What's your view on immigration?"),
        ("alexandria_ocasio_cortez", "How would you address climate change?")
    ]
    
    for persona_id, question in politicians:
        select_persona(persona_id)
        persona = persona_manager.get_active_persona()
        
        print(f"\n{persona['name']}:")
        print(f"Q: {question}")
        
        print("Thinking...")
        response = await run_conversation(question)
        print(f"A: {response}\n")
        print("-" * 40)
    
    print("\nDemo complete! Now try talking to any politician yourself.\n")

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Talk to AI Politicians")
    parser.add_argument("politician", nargs="?", help="Politician to talk to (e.g., bernie_sanders)")
    parser.add_argument("--demo", action="store_true", help="See quick examples")
    parser.add_argument("--list", action="store_true", help="List available politicians")
    args = parser.parse_args()
    
    # Just show the list
    if args.list:
        display_personas()
        return
        
    # Run the demo
    if args.demo:
        await quick_demo()
        persona_id = input("\nWho would you like to talk to? ")
        await chat_with(persona_id)
        return
    
    print_welcome()
    
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

async def run_demo():
    """Run a demonstration with several politicians."""
    print("=== POLITICAL AI DEMONSTRATION ===\n")
    
    # Talk to Bernie Sanders
    print("Bernie Sanders on healthcare:")
    select_persona("bernie_sanders")
    response = await run_conversation("What do you think about healthcare in America?")
    print(f"Bernie Sanders: {response}\n")
    
    # Talk to Donald Trump
    print("Donald Trump on immigration:")
    select_persona("donald_trump")
    response = await run_conversation("What's your opinion on immigration?")
    print(f"Donald Trump: {response}\n")
    
    # Compare perspectives
    print("AOC comparing climate perspectives:")
    select_persona("alexandria_ocasio_cortez")
    response = await run_conversation("Compare different views on climate change")
    print(f"AOC: {response}\n")
    
    # Simulate a debate
    print("Political debate on education:")
    select_persona("jacinda_ardern")
    response = await run_conversation("I'd like to see a debate between politicians on education")
    print(f"Debate: {response}\n")
    
    print("=== DEMO COMPLETE ===")

if __name__ == "__main__":
    asyncio.run(main())

