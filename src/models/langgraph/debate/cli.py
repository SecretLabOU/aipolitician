#!/usr/bin/env python3
"""
Command-line interface for the AI Politician Debate System
=========================================================

This module provides command-line tools to run, configure, and visualize
debates between AI politicians.
"""
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List
import time
from datetime import datetime

# Add the project root to the Python path
root_dir = Path(__file__).parent.parent.parent.parent.parent.absolute()
sys.path.insert(0, str(root_dir))

from src.models.langgraph.debate.workflow import (
    DebateInput, 
    DebateFormat, 
    run_debate
)


def parse_args():
    """Parse command line arguments for the debate system."""
    parser = argparse.ArgumentParser(description="AI Politician Debate System")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Run debate command
    run_parser = subparsers.add_parser("run", help="Run a debate between AI politicians")
    run_parser.add_argument("--topic", type=str, required=True, help="Main debate topic")
    run_parser.add_argument("--participants", type=str, required=True, 
                          help="Comma-separated list of politician identities (e.g., 'biden,trump')")
    run_parser.add_argument("--format", type=str, default="head_to_head", 
                          choices=["town_hall", "head_to_head", "panel"],
                          help="Debate format")
    run_parser.add_argument("--time-per-turn", type=int, default=60,
                          help="Time in seconds allocated per turn")
    run_parser.add_argument("--allow-interruptions", action="store_true",
                          help="Allow interruptions during the debate")
    run_parser.add_argument("--fact-check", action="store_true", default=True,
                          help="Enable fact checking")
    run_parser.add_argument("--moderator-control", type=str, default="moderate",
                          choices=["strict", "moderate", "minimal"],
                          help="Level of moderator control")
    run_parser.add_argument("--no-rag", action="store_true",
                          help="Disable RAG knowledge retrieval")
    run_parser.add_argument("--trace", action="store_true",
                          help="Show trace information during the debate")
    run_parser.add_argument("--output", type=str,
                          help="Output file for debate transcript (JSON)")
    
    # Visualize debate command
    viz_parser = subparsers.add_parser("visualize", help="Visualize the debate workflow")
    
    # Config command to list available politicians
    config_parser = subparsers.add_parser("config", help="Configure debate settings")
    config_parser.add_argument("--list-politicians", action="store_true",
                             help="List available politician identities")
    config_parser.add_argument("--list-formats", action="store_true",
                             help="List available debate formats")
    
    return parser.parse_args()


def format_statement(turn: Dict[str, Any]) -> str:
    """Format a turn's statement for display."""
    speaker = turn["speaker"].upper()
    statement = turn["statement"]
    
    # Format based on whether it's an interruption
    if turn.get("is_interruption", False):
        return f"\n[INTERRUPTION] {speaker}: {statement}\n"
    else:
        return f"\n{speaker}: {statement}\n"


def format_moderator_note(note: Dict[str, Any]) -> str:
    """Format a moderator note for display."""
    message = note["message"]
    
    # Special formatting for topic changes
    if note.get("topic_change", False):
        return f"\n[TOPIC CHANGE] MODERATOR: {message}\n"
    else:
        return f"\nMODERATOR: {message}\n"


def format_fact_check(fact_check: Dict[str, Any]) -> str:
    """Format fact check results for display."""
    speaker = fact_check["speaker"].upper()
    
    output = f"\n[FACT CHECK] Statements by {speaker}:\n"
    
    for claim in fact_check["claims"]:
        accuracy = claim["accuracy"]
        statement = claim["statement"]
        
        # Color-code accuracy
        if accuracy >= 0.8:
            accuracy_str = f"MOSTLY TRUE ({accuracy:.2f})"
        elif accuracy >= 0.6:
            accuracy_str = f"PARTIALLY TRUE ({accuracy:.2f})"
        elif accuracy >= 0.4:
            accuracy_str = f"PARTIALLY FALSE ({accuracy:.2f})"
        else:
            accuracy_str = f"MOSTLY FALSE ({accuracy:.2f})"
        
        output += f"  • Claim: \"{statement}\"\n"
        output += f"    Rating: {accuracy_str}\n"
        
        if claim.get("corrected_info"):
            output += f"    Correction: {claim['corrected_info']}\n"
        
        if claim.get("sources"):
            output += f"    Sources: {', '.join(claim['sources'])}\n"
        
        output += "\n"
    
    return output


def display_debate(debate_result: Dict[str, Any]):
    """Display a debate in a readable format."""
    topic = debate_result["topic"]
    participants = ", ".join(debate_result["participants"])
    
    print(f"\n{'='*80}")
    print(f"DEBATE: {topic}")
    print(f"PARTICIPANTS: {participants}")
    print(f"{'='*80}\n")
    
    # Collect all events (turns, moderator notes, fact checks)
    events = []
    
    # Add turns to events
    for turn in debate_result["turn_history"]:
        events.append({
            "type": "turn",
            "turn": turn["turn"],
            "data": turn,
            "timestamp": turn.get("timestamp", "")
        })
    
    # Add moderator notes to events
    for note in debate_result.get("moderator_notes", []):
        events.append({
            "type": "moderator",
            "turn": note["turn"],
            "data": note,
            "timestamp": note.get("timestamp", "")
        })
    
    # Add fact checks to events
    for check in debate_result.get("fact_checks", []):
        events.append({
            "type": "fact_check",
            "turn": check["turn"],
            "data": check,
            "timestamp": check.get("timestamp", "")
        })
    
    # Sort events by turn number and timestamp
    events.sort(key=lambda e: (e["turn"], e["timestamp"]))
    
    # Display events in order
    for event in events:
        if event["type"] == "turn":
            print(format_statement(event["data"]))
        elif event["type"] == "moderator":
            print(format_moderator_note(event["data"]))
        elif event["type"] == "fact_check":
            print(format_fact_check(event["data"]))
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"DEBATE SUMMARY:")
    print(f"  Topic: {topic}")
    print(f"  Participants: {participants}")
    print(f"  Turns: {len(debate_result['turn_history'])}")
    print(f"  Fact Checks: {len(debate_result.get('fact_checks', []))}")
    print(f"  Subtopics Covered: {', '.join(debate_result.get('subtopics_covered', []))}")
    print(f"{'='*80}\n")


def run_debate_command(args):
    """Run a debate based on command line arguments."""
    # Parse participants list
    participants = [p.strip() for p in args.participants.split(",")]
    
    # Create debate format configuration
    format_config = DebateFormat(
        format_type=args.format,
        time_per_turn=args.time_per_turn,
        allow_interruptions=args.allow_interruptions,
        fact_check_enabled=args.fact_check,
        max_rebuttal_length=250,  # Default value
        moderator_control=args.moderator_control
    )
    
    # Create input configuration
    input_config = DebateInput(
        topic=args.topic,
        format=format_config,
        participants=participants,
        use_rag=not args.no_rag,
        trace=args.trace
    )
    
    print(f"Starting debate on topic: {args.topic}")
    print(f"Participants: {', '.join(participants)}")
    print(f"Format: {args.format} (Interruptions: {'Enabled' if args.allow_interruptions else 'Disabled'}, "
          f"Fact-checking: {'Enabled' if args.fact_check else 'Disabled'})")
    print("Running debate, please wait...\n")
    
    # Run the debate
    start_time = time.time()
    result = run_debate(input_config)
    end_time = time.time()
    
    # Display results
    display_debate(result)
    
    print(f"Debate completed in {end_time - start_time:.2f} seconds.")
    
    # Save to output file if specified
    if args.output:
        with open(args.output, 'w') as f:
            # Add metadata
            result["metadata"] = {
                "generated_at": datetime.now().isoformat(),
                "topic": args.topic,
                "format": args.format,
                "participants": participants,
                "duration_seconds": end_time - start_time
            }
            json.dump(result, f, indent=2)
        print(f"Debate transcript saved to {args.output}")


def visualize_command():
    """Visualize the debate workflow graph."""
    try:
        from langgraph.checkpoint import wandb
        from src.models.langgraph.debate.workflow import create_debate_graph
        
        graph = create_debate_graph()
        
        # Get a unique name for the visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        graph_name = f"debate_graph_{timestamp}"
        
        # Visualize using wandb
        wandb_visual = wandb.visualize(graph, name=graph_name)
        print(f"Graph visualization created: {wandb_visual.url}")
        print("You can view the graph at the URL above.")
        
    except ImportError:
        print("Error: Could not visualize graph. Make sure wandb is installed.")
        print("Try: pip install wandb")


def config_command(args):
    """Handle configuration-related commands."""
    if args.list_politicians:
        try:
            from src.models.langgraph.config import PoliticianIdentity
            
            print("\nAvailable politician identities:")
            for identity in PoliticianIdentity.__members__:
                print(f"  - {identity.lower()}")
            print()
            
        except ImportError:
            print("Error: Could not list politician identities.")
            print("Available politicians: biden, trump (default)")
    
    if args.list_formats:
        print("\nAvailable debate formats:")
        print("  - town_hall: Multiple politicians answering questions, often from the audience")
        print("  - head_to_head: Direct debate between two politicians")
        print("  - panel: Multiple politicians with a moderator leading discussion")
        print()


def main():
    """Main entry point for the debate CLI."""
    args = parse_args()
    
    if args.command == "run":
        run_debate_command(args)
    elif args.command == "visualize":
        visualize_command()
    elif args.command == "config":
        config_command(args)
    else:
        print("Please specify a command. Use --help for usage information.")


if __name__ == "__main__":
    main() 