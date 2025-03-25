#!/usr/bin/env python3
"""
Political Agent System

A streamlined, GPU-accelerated conversation system with political personas.
"""

import os
import sys
import json
import argparse
import asyncio
import logging
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("political_agent")

# Import persona management
from political_agent_graph.persona_manager import (
    get_available_personas,
    set_active_persona,
    get_active_persona
)

# Import conversation functionality
from political_agent_graph.graph import run_conversation

# Initialize console
console = Console()

def list_personas():
    """List all available personas with their descriptions."""
    personas = get_available_personas()
    
    if not personas:
        console.print("[bold red]No personas available.[/bold red]")
        return
    
    console.print(Panel("[bold blue]Available Political Personas[/bold blue]", expand=False))
    
    for persona_id, details in personas.items():
        console.print(f"[bold green]{details['name']} ({persona_id})[/bold green]")
        console.print(f"  Party: {details['party']}")
        console.print(f"  {details.get('description', 'No description available.')}")
        console.print()

async def chat_session(persona_id):
    """Start an interactive chat session with the selected persona."""
    personas = get_available_personas()
    
    if persona_id not in personas:
        console.print(f"[bold red]Error:[/bold red] Persona '{persona_id}' not found.")
        available = ", ".join(personas.keys())
        console.print(f"Available personas: {available}")
        return
    
    # Set the active persona
    set_active_persona(persona_id)
    persona = get_active_persona()
    
    # Welcome message
    console.print(Panel(
        f"[bold]Starting conversation with [blue]{persona['name']}[/blue][/bold]\n"
        f"Type 'exit' or 'quit' to end the conversation.",
        title="Political Agent System",
        expand=False
    ))
    
    # Chat loop
    while True:
        try:
            # Get user input
            user_message = console.input("[bold green]You:[/bold green] ")
            
            # Check for exit command
            if user_message.lower() in ("exit", "quit", "q"):
                console.print("\n[bold]Ending conversation.[/bold]")
                break
                
            # Skip empty messages
            if not user_message.strip():
                continue
                
            # Show thinking indicator
            with console.status("[bold blue]Thinking...[/bold blue]"):
                response = await run_conversation(user_message)
                
            # Display response
            console.print(f"[bold red]{persona['name']}:[/bold red] {response}")
            console.print()
            
        except KeyboardInterrupt:
            console.print("\n[bold]Conversation interrupted.[/bold]")
            break
        except Exception as e:
            logger.exception("Error in conversation")
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
            console.print("Try asking a different question.")

async def quick_demo():
    """Run a quick demo with preset questions for each persona."""
    personas = get_available_personas()
    
    if not personas:
        console.print("[bold red]No personas available for demo.[/bold red]")
        return
    
    # Demo questions
    questions = [
        "What's your stance on climate change?",
        "How would you handle the economy?",
        "What are your thoughts on healthcare reform?"
    ]
    
    console.print(Panel("[bold]Political Agent Demo[/bold]", expand=False))
    
    # Run through each persona
    for persona_id, details in list(personas.items())[:2]:  # Limit to first 2 personas for brevity
        console.print(f"\n[bold blue]{details['name']} ({details['party']})[/bold blue]")
        set_active_persona(persona_id)
        
        # Ask each question
        for question in questions:
            console.print(f"[bold green]Question:[/bold green] {question}")
            
            with console.status("[bold]Generating response...[/bold]"):
                response = await run_conversation(question)
                
            console.print(f"[bold red]Response:[/bold red] {response}")
            console.print()
            
            # Small delay between questions
            await asyncio.sleep(1)
        
        # Separator between personas
        console.print("=" * 50)
        await asyncio.sleep(2)

async def main():
    """Main entry point for the program."""
    parser = argparse.ArgumentParser(description="Political Agent System")
    
    # Create mutually exclusive group for the main actions
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--chat", "-c", metavar="PERSONA", 
                      help="Start chat with persona (e.g., trump, biden)")
    group.add_argument("--demo", "-d", action="store_true", 
                      help="Run demonstration")
    group.add_argument("--list", "-l", action="store_true", 
                      help="List available personas")
    
    args = parser.parse_args()
    
    try:
        if args.list:
            list_personas()
        elif args.demo:
            await quick_demo()
        elif args.chat:
            await chat_session(args.chat)
    except KeyboardInterrupt:
        console.print("\n[bold]Program interrupted by user.[/bold]")
    except Exception as e:
        logger.exception("Unexpected error")
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))