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
import os
import traceback

# Add the project root to the Python path
root_dir = Path(__file__).parent.parent.parent.parent.parent.absolute()
sys.path.insert(0, str(root_dir))

from src.models.langgraph.debate.workflow import (
    DebateInput, 
    DebateFormat, 
    run_debate,
    run_simplified_debate
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
    run_parser.add_argument("--no-fact-check", action="store_true",
                          help="Disable fact checking (overrides --fact-check)")
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
        
        # Use rating directly if available, or determine based on accuracy score
        if "rating" in claim:
            rating_label = claim["rating"]
        else:
            # Color-code accuracy based on score ranges
            if accuracy >= 0.95:
                rating_label = "TRUE"
            elif accuracy >= 0.80:
                rating_label = "MOSTLY TRUE"
            elif accuracy >= 0.60:
                rating_label = "PARTIALLY TRUE"
            elif accuracy >= 0.40:
                rating_label = "MIXED"
            elif accuracy >= 0.20:
                rating_label = "PARTIALLY FALSE"
            elif accuracy >= 0.05:
                rating_label = "MOSTLY FALSE"
            else:
                rating_label = "FALSE"
        
        output += f"  • Claim: \"{statement}\"\n"
        output += f"    Rating: {rating_label} ({accuracy:.2f})\n"
        
        if claim.get("corrected_info"):
            output += f"    {claim['corrected_info']}\n"
        
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
    seen_statements = set()  # Track statements to avoid duplicates
    seen_topic_changes = set()  # Track topic changes to avoid duplicates
    
    # First, identify the introduction event to ensure it comes first
    introduction_event = None
    for note in debate_result.get("moderator_notes", []):
        if note.get("is_introduction", False) or (note["turn"] == 0 and "Welcome to today's" in note.get("message", "")):
            introduction_event = {
                "type": "moderator",
                "turn": -1,  # Ensure it comes first
                "data": note,
                "timestamp": "0"  # Ensure it comes first
            }
            break
    
    # If we found an introduction, add it first
    if introduction_event:
        events.append(introduction_event)
    
    # Add turns to events - avoid duplicates
    for turn in debate_result["turn_history"]:
        # Create a simplified version of the statement for duplicate detection
        statement_key = f"{turn['speaker']}:{turn['statement'][:50]}"
        
        # Only add if we haven't seen this statement before
        if statement_key not in seen_statements:
            events.append({
                "type": "turn",
                "turn": turn["turn"],
                "data": turn,
                "timestamp": turn.get("timestamp", "")
            })
            seen_statements.add(statement_key)
    
    # Add remaining moderator notes - avoid duplicates and ensure proper order
    for note in debate_result.get("moderator_notes", []):
        # Skip the introduction as we've already added it
        if note.get("is_introduction", False) or (note["turn"] == 0 and "Welcome to today's" in note.get("message", "")):
            continue
            
        # Check for topic change notes to avoid duplicates
        if "topic_change" in note and note.get("topic_change", False):
            topic_key = note.get("new_topic", "")
            
            # Skip if we've already seen this topic change
            if topic_key and topic_key in seen_topic_changes:
                continue
                
            if topic_key:
                seen_topic_changes.add(topic_key)
        
        # Include this note
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
    
    # Sort events by turn number and timestamp (except introduction which is fixed at the start)
    events.sort(key=lambda e: (-100 if e.get("turn", 0) == -1 else e.get("turn", 0), e.get("timestamp", "")))
    
    # Keep track of last speaker for better formatting
    last_speaker = None
    
    # Display events in logical order
    for event in events:
        # Skip empty turns or duplicate content
        if event["type"] == "turn" and not event["data"].get("statement"):
            continue
            
        # For turns, track the speaker to avoid showing "Your time is up" right after a speaker
        # just started (which can happen with moderator notes out of order)
        if event["type"] == "turn":
            last_speaker = event["data"]["speaker"]
        
        # For moderator notes, check if it's a transition to the same speaker who just talked
        if (event["type"] == "moderator" and 
            "Your time is up" in event["data"].get("message", "") and 
            last_speaker == event["data"].get("next_speaker")):
            continue
        
        # Display the event
        if event["type"] == "turn":
            print(format_statement(event["data"]))
        elif event["type"] == "moderator":
            print(format_moderator_note(event["data"]))
        elif event["type"] == "fact_check":
            print(format_fact_check(event["data"]))
    
    # Calculate subtopics covered
    subtopics = set()
    for turn in debate_result["turn_history"]:
        if "subtopic" in turn and turn["subtopic"]:
            subtopics.add(turn["subtopic"])
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"DEBATE SUMMARY:")
    print(f"  Topic: {topic}")
    print(f"  Participants: {participants}")
    print(f"  Turns: {len(debate_result['turn_history'])}")
    print(f"  Fact Checks: {len(debate_result.get('fact_checks', []))}")
    
    # Format subtopics nicely
    if subtopics:
        print(f"  Subtopics Covered: {', '.join(sorted(subtopics))}")
    else:
        print(f"  Subtopics Covered: Main topic only")
    
    print(f"{'='*80}\n")


def format_debate_output(debate_result: Dict[str, Any]) -> str:
    """Format the debate result into a readable string."""
    try:
        # Check if the debate result is already a string
        if isinstance(debate_result, str):
            return debate_result
            
        # Extract basic information
        topic = debate_result.get("topic", "Unknown Topic")
        participants = debate_result.get("participants", ["Unknown Participants"])
        if isinstance(participants, list):
            participants_str = ", ".join(participants)
        else:
            participants_str = str(participants)
            
        # Build the header
        output = f"""
================================================================================
DEBATE: {topic}
PARTICIPANTS: {participants_str}
================================================================================

"""
        
        # Add moderator introduction if available
        intro_notes = [note for note in debate_result.get("moderator_notes", []) 
                       if note.get("turn", -1) == 0]
        if intro_notes:
            output += f"MODERATOR: {intro_notes[0].get('message', 'Welcome to the debate.')}\n\n"
        
        # Process turns and moderator notes chronologically
        turn_history = debate_result.get("turn_history", [])
        moderator_notes = debate_result.get("moderator_notes", [])
        
        # Combine events and sort by turn number
        events = []
        
        # Add turn events
        for turn in turn_history:
            events.append({
                "type": "turn",
                "turn": turn.get("turn", 0),
                "content": f"{turn.get('speaker', 'Unknown').upper()}: {turn.get('statement', '')}\n\n"
            })
        
        # Add moderator notes (excluding intro which was already added)
        for note in moderator_notes:
            if note.get("turn", -1) == 0:  # Skip intro
                continue
                
            # Check if it's a topic change
            if note.get("topic_change", False):
                prefix = "[TOPIC CHANGE] "
            else:
                prefix = ""
                
            events.append({
                "type": "moderator",
                "turn": note.get("turn", 0),
                "content": f"{prefix}MODERATOR: {note.get('message', '')}\n\n"
            })
        
        # Add fact checks
        for fact_check in debate_result.get("fact_checks", []):
            speaker = fact_check.get("speaker", "Unknown")
            claims = fact_check.get("claims", [])
            
            # Start building the fact check content
            fact_content = f"[FACT CHECK] Claims by {speaker.upper()}:\n"
            
            # Process each claim
            if claims and isinstance(claims[0], dict):
                # Old format with individual accuracy per claim
                for claim in claims:
                    statement = claim.get("statement", "")
                    accuracy = claim.get("accuracy", 0)
                    rating = claim.get("rating", "UNKNOWN")
                    
                    fact_content += f"  • \"{statement}\"\n"
                    fact_content += f"    Rating: {rating} ({int(accuracy * 100)}% accurate)\n"
            else:
                # New format with claims as strings and single accuracy
                for i, claim in enumerate(claims):
                    fact_content += f"  • Claim {i+1}: \"{claim}\"\n"
                
                accuracy = fact_check.get("accuracy", 0)
                rating = fact_check.get("rating", "UNKNOWN")
                fact_content += f"  Rating: {rating} ({int(accuracy * 100)}% accurate)\n"
            
            fact_content += "\n"
            
            # Add to events
            events.append({
                "type": "fact_check",
                "turn": fact_check.get("turn", 0),
                "content": fact_content
            })
        
        # Sort events by turn number
        events.sort(key=lambda e: e.get("turn", 0))
        
        # Add all events to the output
        for event in events:
            output += event.get("content", "")
        
        # Add closing
        output += f"""
================================================================================
DEBATE SUMMARY:
  Topic: {topic}
  Participants: {participants_str}
  Turns: {len(turn_history)}
  Fact Checks: {len(debate_result.get("fact_checks", []))}
================================================================================
"""
        
        return output
        
    except Exception as e:
        # If anything goes wrong, return a basic formatted string
        return f"Error formatting debate: {e}\nRaw debate data: {str(debate_result)[:500]}..."


def run_command(args):
    """Run a debate session with the given arguments."""
    try:
        # If trace is enabled, set it in the environment
        if args.trace:
            os.environ["DEBATE_TRACE"] = "1"
            
        # Parse topic
        topic = args.topic
        
        # Parse participants - enhanced error checking
        participants_raw = args.participants
        if '--' in participants_raw:
            print(f"WARNING: Possible malformed command. Found '--' in participants: '{participants_raw}'")
            print("Did you mean to add a space before the flag? Example: 'biden,trump' --trace")
            # Try to fix it by splitting at the first '--'
            participants_raw = participants_raw.split('--')[0]
            print(f"Using participants: '{participants_raw}'")
            
        participants = [p.strip() for p in participants_raw.split(",") if p.strip()]
        
        # Validate we have at least one valid participant
        if not participants:
            participants = ["biden", "trump"]
            print(f"WARNING: No valid participants found in '{args.participants}', using defaults: {participants}")
        
        # Parse format
        format_name = args.format.lower() if args.format else "head_to_head"
        
        # Determine whether to use RAG
        use_rag = not args.no_rag if hasattr(args, 'no_rag') else True
        
        # Determine if interruptions are enabled
        interruptions_enabled = args.allow_interruptions if hasattr(args, 'allow_interruptions') else False
        
        # Determine if fact checking should be enabled (--no-fact-check overrides --fact-check)
        fact_check_enabled = not args.no_fact_check if hasattr(args, 'no_fact_check') else args.fact_check
        
        # Set up input for debate
        debate_input = DebateInput(
            topic=topic,
            participants=participants,
            format=DebateFormat(
                name=format_name,
                fact_check_enabled=fact_check_enabled,
                interruptions_enabled=interruptions_enabled
            ),
            trace=args.trace,
            use_rag=use_rag
        )
        
        # Measure execution time
        start_time = time.time()
        
        # Run the debate
        print(f"Starting debate on topic: {topic}")
        print(f"Participants: {', '.join(participants)}")
        print(f"Format: {format_name} (Interruptions: {'Enabled' if interruptions_enabled else 'Disabled'}, Fact-checking: {'Enabled' if fact_check_enabled else 'Disabled'})")
        print("Running debate, please wait...\n")
        
        try:
            # Try the main LangGraph workflow first
            result = run_debate(debate_input)
        except Exception as e:
            # If it fails, fall back to simplified debate
            print(f"\nEncountered an error with the LangGraph workflow: {e}")
            print("Running simplified debate mode...\n")
            result = run_simplified_debate(debate_input.model_dump())
        
        # End time measurement
        elapsed_time = time.time() - start_time
        
        # Print the result
        if isinstance(result, str):
            # If the result is already a formatted string (from simplified_debate)
            print(result)
        else:
            # Try to format the debate from the state object
            formatted_debate = format_debate_output(result)
            print(formatted_debate)
            
            # Display fact check summary at the end if any fact checks were performed
            fact_checks = result.get("fact_checks", [])
            if fact_checks:
                print("\n================================================================================")
                print(f"FACT CHECK SUMMARY: {len(fact_checks)} claims verified")
                print("================================================================================")
                
                for i, check in enumerate(fact_checks):
                    speaker = check.get("speaker", "Unknown")
                    claims = check.get("claims", [])
                    
                    if isinstance(claims, list) and claims:
                        print(f"\nFact Check #{i+1} - Speaker: {speaker}")
                        
                        if isinstance(claims[0], dict):
                            # Old format with multiple claim dictionaries
                            for j, claim_data in enumerate(claims):
                                statement = claim_data.get("statement", "Unknown claim")
                                accuracy = claim_data.get("accuracy", 0)
                                rating = claim_data.get("rating", "UNKNOWN")
                                
                                print(f"  Claim: \"{statement}\"")
                                print(f"  Rating: {rating} ({int(accuracy * 100)}% accurate)")
                                print("")
                        else:
                            # New format with claims as strings and a single accuracy/rating
                            for j, claim in enumerate(claims):
                                print(f"  Claim {j+1}: \"{claim}\"")
                            
                            accuracy = check.get("accuracy", 0)
                            rating = check.get("rating", "UNKNOWN")
                            print(f"  Rating: {rating} ({int(accuracy * 100)}% accurate)")
                            print("")
                
                print("================================================================================")
                
        print(f"Debate completed in {elapsed_time:.2f} seconds.")
        
    except Exception as e:
        print(f"Error running debate: {e}")
        traceback.print_exc()


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
        run_command(args)
    elif args.command == "visualize":
        visualize_command()
    elif args.command == "config":
        config_command(args)
    else:
        print("Please specify a command. Use --help for usage information.")


if __name__ == "__main__":
    main() 