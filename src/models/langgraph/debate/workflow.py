#!/usr/bin/env python3
"""
LangGraph Debate System for AI Politicians
==========================================

This module defines a workflow that enables multiple AI politicians to engage
in structured debates with each other, including turn-taking, topic management,
fact-checking, and rebuttals.
"""
import sys
from pathlib import Path
from typing import Dict, Any, TypedDict, List, Literal, Optional, Union
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from datetime import datetime
import random

# Add the project root to the Python path
root_dir = Path(__file__).parent.parent.parent.parent.parent.absolute()
sys.path.insert(0, str(root_dir))

from src.models.langgraph.config import PoliticianIdentity
from src.models.langgraph.debate.agents import (
    moderate_debate,
    politician_turn,
    fact_check,
    handle_interruption,
    manage_topic,
    generate_introduction,
    generate_transition,
    generate_subtopics
)


# Define debate formats
class DebateFormat(BaseModel):
    """Configuration for the debate format."""
    name: Literal["town_hall", "head_to_head", "panel"] = Field(..., 
                                                         description="Type of debate format")
    time_per_turn: int = Field(default=60, description="Time in seconds allocated per turn")
    interruptions_enabled: bool = Field(default=True, description="Whether interruptions are allowed")
    fact_check_enabled: bool = Field(default=True, description="Whether fact checking is enabled")
    max_rebuttal_length: int = Field(default=250, description="Maximum character length for rebuttals")
    moderator_control: Literal["strict", "moderate", "minimal"] = Field(default="moderate", 
                                                            description="Level of moderator control")


# Define input/output schemas
class DebateInput(BaseModel):
    """Input schema for the AI Politician debate workflow."""
    topic: str = Field(..., description="Main debate topic")
    format: DebateFormat = Field(..., description="Debate format configuration")
    participants: List[str] = Field(..., description="List of politician identities participating")
    opening_statement: Optional[str] = Field(None, description="Optional opening statement or question")
    use_rag: bool = Field(default=True, description="Whether to use RAG for knowledge retrieval")
    trace: bool = Field(default=False, description="Whether to output trace information")


class DebaterState(BaseModel):
    """State for an individual debater."""
    identity: str = Field(..., description="Politician identity")
    position: str = Field(default="", description="Current position on the topic")
    knowledge: str = Field(default="", description="Retrieved knowledge relevant to the topic")
    pending_rebuttal: str = Field(default="", description="Pending rebuttal to another politician")
    interrupted: bool = Field(default=False, description="Whether this politician has been interrupted")


class FactCheck(BaseModel):
    """Fact check result."""
    statement: str = Field(..., description="Statement being fact-checked")
    accuracy: float = Field(..., description="Accuracy score from 0.0 to 1.0")
    corrected_info: Optional[str] = Field(None, description="Corrected information if inaccurate")
    sources: List[str] = Field(default_factory=list, description="Sources supporting the fact check")


# State type for the debate workflow
class DebateState(TypedDict):
    topic: str
    format: Dict[str, Any]
    participants: List[str]
    use_rag: bool
    trace: bool
    current_speaker: str
    speaking_queue: List[str]
    debater_states: Dict[str, Dict[str, Any]]
    turn_history: List[Dict[str, Any]]
    fact_checks: List[Dict[str, Any]]
    moderator_notes: List[str]
    interruption_requested: bool
    current_subtopic: str


def create_debate_graph() -> StateGraph:
    """Create the LangGraph workflow for the AI Politician debate."""
    # Initialize the state graph with the appropriate state type
    workflow = StateGraph(DebateState)
    
    # Add nodes for each component of the debate
    workflow.add_node("initialize_debate", initialize_debate)
    workflow.add_node("moderator", moderate_debate)
    workflow.add_node("debater", politician_turn)
    workflow.add_node("fact_checker", fact_check)
    workflow.add_node("interruption_handler", handle_interruption)
    workflow.add_node("topic_manager", manage_topic)
    
    # Define the edges (flow) of the graph
    workflow.set_entry_point("initialize_debate")
    
    # Initialize debate -> Moderator introduction
    workflow.add_edge("initialize_debate", "moderator")
    
    # Define a conditional check for maximum turns (clearer end condition)
    def should_end(state: DebateState) -> bool:
        """Check if the debate should end based on number of turns."""
        return len(state.get("turn_history", [])) >= 10
    
    # More explicit end condition check for LangGraph 0.3.8
    def determine_next_step(state: DebateState) -> str:
        """Determine the next step with clearer end condition."""
        if should_end(state):
            return "end"
        elif len(state.get("turn_history", [])) % 4 == 0 and len(state.get("turn_history", [])) > 0:
            return "topic_manager"
        else:
            return "debater"
    
    # Conditional routing from moderator with explicit end condition
    workflow.add_conditional_edges(
        "moderator",
        determine_next_step,
        {
            "debater": "debater",
            "topic_manager": "topic_manager",
            "end": END
        }
    )
    
    # Ensure fact_checker can also end the debate if max turns reached
    workflow.add_conditional_edges(
        "fact_checker",
        lambda state: "end" if should_end(state) else "moderator",
        {
            "moderator": "moderator",
            "end": END
        }
    )
    
    # Ensure interruption_handler can also end the debate if max turns reached
    workflow.add_conditional_edges(
        "interruption_handler",
        lambda state: "end" if should_end(state) else "moderator",
        {
            "moderator": "moderator",
            "end": END
        }
    )
    
    # Ensure topic_manager can also end the debate if max turns reached
    workflow.add_conditional_edges(
        "topic_manager",
        lambda state: "end" if should_end(state) else "moderator",
        {
            "moderator": "moderator",
            "end": END
        }
    )
    
    # The debater node should check for interruptions or fact checks
    # but should also be able to end the debate if max turns reached
    workflow.add_conditional_edges(
        "debater",
        lambda state: "end" if should_end(state) else check_interruption(state),
        {
            "interruption": "interruption_handler",
            "fact_check": "fact_checker", 
            "moderator": "moderator",
            "end": END
        }
    )
    
    return workflow


def initialize_debate(state: DebateState) -> DebateState:
    """Initialize the debate state and setup."""
    # Implementation with trace info if enabled
    if state.get("trace", False):
        print("\n🔎 TRACE: Initializing Debate")
        print("=================================")
        print(f"Topic: {state['topic']}")
        print(f"Format: {state.get('format', {}).get('name', 'unknown')}")
        print(f"Participants: {', '.join(state.get('participants', []))}")
        print("---------------------------------")
    
    # Defensive programming - ensure all required fields exist
    result = state.copy()
    
    # Validate that we have the minimum required fields
    if not result.get("topic"):
        print("Warning: Topic not found in input state, setting to 'General Debate'")
        result["topic"] = "General Debate"
        
    if not result.get("participants"):
        print("Warning: Participants not found in input state, defaulting to ['biden', 'trump']")
        result["participants"] = ["biden", "trump"]
    
    # Ensure format is properly initialized
    if not result.get("format"):
        print("Warning: Format not found in input state, using default head_to_head format")
        result["format"] = {
            "name": "head_to_head",
            "time_per_turn": 60,
            "interruptions_enabled": True,
            "fact_check_enabled": True,
            "max_rebuttal_length": 250,
            "moderator_control": "moderate"
        }
    
    # Set up initial speaking queue based on format
    result["speaking_queue"] = state.get("participants", []).copy()
    
    # Avoid empty speaking queue
    if not result["speaking_queue"]:
        print("Warning: Empty speaking queue, using default participants")
        result["speaking_queue"] = ["biden", "trump"]
    
    # Set the current speaker
    result["current_speaker"] = result["speaking_queue"].pop(0)
    
    # Set current subtopic (default to main topic)
    result["current_subtopic"] = state.get("current_subtopic", state["topic"])
    
    # Initialize states for each debater
    result["debater_states"] = {}
    for participant in result["participants"]:
        # Check if debater state already exists
        if participant in state.get("debater_states", {}):
            result["debater_states"][participant] = state["debater_states"][participant]
        else:
            # Create new debater state
            result["debater_states"][participant] = {
                "identity": participant,
                "position": "",
                "knowledge": "",
                "pending_rebuttal": "",
                "interrupted": False
            }
    
    # Initialize other tracking fields (preserving existing values if present)
    result["turn_history"] = state.get("turn_history", [])
    result["fact_checks"] = state.get("fact_checks", [])
    result["moderator_notes"] = state.get("moderator_notes", [])
    result["interruption_requested"] = state.get("interruption_requested", False)
    result["use_rag"] = state.get("use_rag", True)
    
    # Add a starting timestamp if not present
    if not result.get("debate_start_time"):
        result["debate_start_time"] = datetime.now().isoformat()
    
    return result


def next_step(state: DebateState) -> str:
    """Determine the next step in the debate workflow.
    
    Note: This function is kept for backwards compatibility, but newer code uses 
    the determine_next_step function directly in create_debate_graph.
    """
    try:
        # Check if debate should end due to turn limit
        if len(state.get("turn_history", [])) >= 10:
            return "end"
        
        # Check if topic needs management
        if len(state.get("turn_history", [])) % 4 == 0 and len(state.get("turn_history", [])) > 0:
            return "topic_manager"
        
        # Default to next debater
        return "debater"
    except Exception as e:
        # If any error occurs, try to continue with debater or end if too many turns
        print(f"Error in next_step: {e}")
        if len(state.get("turn_history", [])) >= 8:
            return "end"
        return "debater"


def check_interruption(state: DebateState) -> str:
    """Check if an interruption should occur or if we should move to fact checking.
    
    Note: This function is now just responsible for determining the path after
    the debater node, not for deciding when to end the debate (which is handled
    separately with should_end).
    """
    # First check for interruption
    if state.get("interruption_requested", False) and state.get("format", {}).get("interruptions_enabled", False):
        return "interruption"
    
    # Then check if fact checking is needed
    if state.get("format", {}).get("fact_check_enabled", False) and len(state.get("turn_history", [])) > 0:
        return "fact_check"
    
    # Return to moderator for next turn
    return "moderator"


def run_debate(input_data: DebateInput) -> Dict[str, Any]:
    """
    Run a debate between AI politicians.
    
    Args:
        input_data: Debate configuration and topic
        
    Returns:
        Dict[str, Any]: The debate results including all turns and fact checks
    """
    # Create the graph
    graph = create_debate_graph()
    
    # Check LangGraph version with minimal output
    try:
        import langgraph
        import pkg_resources
        import logging
        
        # Configure minimal logging
        logging.basicConfig(level=logging.ERROR)
        
        # Detect version silently
        langgraph_version = pkg_resources.get_distribution("langgraph").version
    except Exception:
        langgraph_version = "unknown"
    
    # Create initial state
    initial_state: DebateState = {
        "topic": input_data.topic,
        "format": input_data.format.model_dump() if hasattr(input_data.format, "model_dump") else input_data.format.dict(),
        "participants": input_data.participants,
        "use_rag": input_data.use_rag,
        "trace": input_data.trace,
        "current_speaker": "",
        "speaking_queue": [],
        "debater_states": {},
        "turn_history": [],
        "fact_checks": [],
        "moderator_notes": [],
        "interruption_requested": False,
        "current_subtopic": input_data.topic
    }
    
    # Try to run the debate using LangGraph with minimal output
    try:
        # Minimal status message
        print("Loading politician response models...")
        
        # Configure for LangGraph 0.3.8 specifically
        if langgraph_version == "0.3.8":
            # Try to run with 0.3.8-specific approach (without verbose output)
            try:
                debate_chain = graph.compile()
                result = debate_chain.invoke(
                    initial_state,
                    {"debug": False, "recursion_limit": 100}
                )
                return result
            except Exception:
                # Silently try alternative method
                try:
                    result = debate_chain.invoke(initial_state, debug=False, recursion_limit=100)
                    return result
                except Exception:
                    # Continue to other approaches
                    pass
        
        # Try other approaches without verbose logging
        try:
            # Try different compilation methods silently
            try:
                from langgraph.prebuilt import DispatchConfig
                config = DispatchConfig(recursion_limit=100)
                debate_chain = graph.compile(config=config)
            except ImportError:
                try:
                    debate_chain = graph.compile(recursion_limit=100, checkpointer=None)
                except Exception:
                    try:
                        from langgraph.checkpoint import NullCheckpointer
                        debate_chain = graph.compile(recursion_limit=100, checkpointer=NullCheckpointer())
                    except Exception:
                        debate_chain = graph.compile()
            
            # Try to run the debate
            result = debate_chain.invoke(initial_state)
            return result
            
        except Exception:
            # Fall back to manual simulation
            pass
    except Exception:
        # Silently handle errors
        pass
    
    # If we get here, all LangGraph approaches failed - run simplified debate
    print("Running simplified debate mode...")
    return run_simplified_debate(initial_state)


def run_simplified_debate(state: DebateState) -> DebateState:
    """
    Run a simplified debate with alternating speakers and fixed turns.
    This is a fallback when the more complex workflow fails.
    """
    print("Running simplified debate...")
    print("Initializing debate state...")
    
    if state.get("trace", False):
        print("\n🔎 TRACE: Initializing Debate")
        print("=================================")
        print(f"Topic: {state['topic']}")
        print(f"Format: {state.get('format', {}).get('name', 'head_to_head')}")
        print(f"Participants: {', '.join(state.get('participants', []))}")
        print("---------------------------------")

    # Validate participants
    if not state.get('participants') or not isinstance(state.get('participants'), list) or '--' in str(state.get('participants')):
        # Something is wrong with participants, try to fix it
        raw_participants = str(state.get('participants', 'biden,trump'))
        if '--' in raw_participants:
            # Handle case where command flags are merged with participants
            print(f"WARNING: Detected possible command flag in participants: {raw_participants}")
            raw_participants = raw_participants.split('--')[0]
            print(f"Fixed participants: {raw_participants}")
        
        # Parse the fixed participants string
        state['participants'] = [p.strip() for p in raw_participants.split(',') if p.strip()]
        
        # Fallback if still empty
        if not state['participants']:
            state['participants'] = ["biden", "trump"]
            print("Using default participants: biden, trump")
    
    # Create a moderator introduction
    moderator_intro = f"Welcome to today's {state.get('format', {}).get('name', 'head_to_head')} debate on the topic of '{state['topic']}'. " \
                     f"Participating in this debate are {', '.join(state['participants'])}. " \
                     f"Each speaker will have 60 seconds per turn. " \
                     f"{'No interruptions will be permitted.' if not state.get('format', {}).get('interruptions_enabled', False) else 'Interruptions may occur.'} " \
                     f"{'Statements will be fact-checked for accuracy.' if state.get('format', {}).get('fact_check_enabled', True) else 'Statements will not be fact-checked.'} " \
                     f"Let's begin with {state['participants'][0]}."
    
    debate_output = [
        f"\n\nMODERATOR: {moderator_intro}\n\n"
    ]
    
    # Initialize turn history if not present
    if 'turn_history' not in state:
        state['turn_history'] = []
    
    # Track fact checks
    fact_checks_count = 0
    
    # Change: Fix maximum turns from 6 to 8 for better depth
    max_turns = 8
    print(f"Topic: {state['topic']}")
    print(f"Participants: {', '.join(state['participants'])}")
    print(f"Starting debate with moderator introduction...")
    
    # Change: Define speaking queue explicitly
    speaking_queue = state['participants'] * (max_turns // len(state['participants']) + 1)
    
    # Track current subtopic for transitions
    current_subtopic = state['topic']
    covered_subtopics = set([current_subtopic])
    
    # Counter for turns with the same subtopic
    same_subtopic_turns = 0
    
    # Run the debate with alternating speakers
    for turn in range(1, max_turns + 1):
        current_speaker = speaking_queue[turn - 1]
        print(f"Turn {turn}/{max_turns}: {current_speaker}...")
        
        # Update state with current speaker
        state['current_speaker'] = current_speaker
        
        # Generate the politician's response
        if state.get("trace", False):
            print(f"\n🔎 TRACE: Politician Agent")
            print("============================")
            print(f"Current Speaker: {current_speaker}")
            print(f"Turn: {turn-1}")
            print("----------------------------")
        
        # Handle subtopic changes every 4 turns (was 3)
        if turn % 4 == 0 and state.get('format', {}).get('name', 'head_to_head') in ['head_to_head', 'town_hall']:
            same_subtopic_turns += 1
            
            if state.get("trace", False):
                print(f"\n🔎 TRACE: Topic Manager")
                print("=========================")
                print(f"Current Topic: {state['topic']}")
                print(f"Current Subtopic: {current_subtopic}")
                print(f"Turn: {turn}")
                print("-------------------------")
                
            # If we've had 2 turns on the same subtopic, try to change it
            if same_subtopic_turns >= 2:
                # Generate subtopics only once to avoid unnecessary changes
                if 'subtopics' not in state:
                    # Generate 3-5 subtopics
                    state['subtopics'] = [
                        f"{state['topic']} - Economic Impact",
                        f"{state['topic']} - Social Implications",
                        f"{state['topic']} - Historical Context",
                        f"{state['topic']} - Future Outlook",
                        f"{state['topic']} - Policy Reform",
                        f"{state['topic']} - International Perspective"
                    ]
                    random.shuffle(state['subtopics'])
                
                # Find a subtopic we haven't covered yet
                new_subtopic = None
                for subtopic in state['subtopics']:
                    if subtopic not in covered_subtopics:
                        new_subtopic = subtopic
                        break
                        
                # If all subtopics covered, pick a random one
                if not new_subtopic and state['subtopics']:
                    new_subtopic = random.choice(state['subtopics'])
                
                if new_subtopic and new_subtopic != current_subtopic:
                    current_subtopic = new_subtopic
                    covered_subtopics.add(current_subtopic)
                    same_subtopic_turns = 0
                    print(f"Topic changed to: {current_subtopic}")
                    
                    # Add a moderator transition to indicate the topic change
                    debate_output.append(f"\n\n[TOPIC CHANGE] MODERATOR: Let's move on to discuss {current_subtopic}.\n\n")
                
        # Generate response based on context and politician identity
        statement = generate_politician_response(
            topic=state['topic'] if current_subtopic == state['topic'] else current_subtopic,
            politician_name=current_speaker,
            debate_context="\n".join(debate_output[-3:]) if len(debate_output) > 0 else "",
            opponent_names=[p for p in state['participants'] if p != current_speaker],
            previous_statements=[t.get('statement', '') for t in state['turn_history'][-3:]] if 'turn_history' in state else []
        )
        
        # Add the statement to the turn history
        turn_data = {
            'turn': turn,
            'speaker': current_speaker,
            'statement': statement,
            'subtopic': current_subtopic,
            'timestamp': datetime.now().isoformat()
        }
        state['turn_history'].append(turn_data)
        
        # Update the latest statement for fact-checking
        state['latest_statement'] = statement
        
        # Add the speaker's statement to the debate output FIRST (before fact checking)
        debate_output.append(f"{current_speaker.upper()}: {statement}\n\n")
        
        # Check facts if enabled
        if state.get('format', {}).get('fact_check_enabled', True):
            if state.get("trace", False):
                print(f"\n🔎 TRACE: Fact Checker Agent")
                print("=============================")
                print(f"Checking statement by: {current_speaker}")
                print(f"Statement length: {len(statement)} chars")
                print("-----------------------------")
            
            # Run fact checking on the statement
            fact_check_results = fact_check(state)
            state.update(fact_check_results)
            
            # Add fact check to the output AFTER the statement if any were generated
            if fact_check_results.get("latest_fact_check"):
                latest_check = fact_check_results["latest_fact_check"]
                fact_checks_count += 1
                
                # Format the fact check for display
                accuracy_pct = int(latest_check["accuracy"] * 100)
                fact_check_text = f"\n\n[FACT CHECK] Claims by {current_speaker}:\n"
                
                for i, claim in enumerate(latest_check["claims"]):
                    fact_check_text += f"Claim {i+1}: \"{claim}\"\n"
                
                fact_check_text += f"Rating: {latest_check['rating']} ({accuracy_pct}% accurate)\n\n"
                debate_output.append(fact_check_text)
        
        # Add moderator transition
        debate_output.append(f"MODERATOR: Your time is up. {'Next up is ' + speaking_queue[turn] + '.' if turn < max_turns else 'This concludes our debate.'}\n\n")
        
        if state.get("trace", False):
            print(f"\n🔎 TRACE: Moderator Agent")
            print("===========================")
            print(f"Turn: {turn}")
            print(f"Current Speaker: {current_speaker}")
            print(f"Current Subtopic: {current_subtopic}")
            print("---------------------------")
    
    print(f"Reached maximum turns ({max_turns}), ending debate")
    print(f"Debate completed: {max_turns} turns, {fact_checks_count} fact checks")
    
    # Combine all debate output and return
    full_debate = "".join(debate_output)
    
    # Format the debate with a header and footer
    formatted_debate = f"""
================================================================================
DEBATE: {state['topic']}
PARTICIPANTS: {', '.join(state['participants'])}
================================================================================

{full_debate}
================================================================================
DEBATE SUMMARY:
  Topic: {state['topic']}
  Participants: {', '.join(state['participants'])}
  Turns: {max_turns}
  Fact Checks: {fact_checks_count}
  Subtopics Covered: {', '.join(list(covered_subtopics)[:3])}
================================================================================
"""
    
    return formatted_debate 

def generate_politician_response(
    topic: str,
    politician_name: str,
    debate_context: str = "",
    opponent_names: List[str] = None,
    previous_statements: List[str] = None
) -> str:
    """
    Generate a response from a politician based on their identity and the debate context.
    This is a simplified version of the response generation for the simplified debate flow.
    """
    from src.models.langgraph.agents.response_agent import generate_response

    # Default empty collections if None
    opponent_names = opponent_names or []
    previous_statements = previous_statements or []
    
    # Build context from previous statements and opponents
    opponents_text = ", ".join(opponent_names) if opponent_names else "opponents"
    
    # Prepare the input state for the response agent
    input_state = {
        "user_input": topic,
        "politician_identity": politician_name,
        "context": f"""
Topic: {topic}
Previous context: {debate_context}
You are {politician_name} participating in a debate against {opponents_text}.
Respond to the debate topic and any relevant points raised by your opponents.
Keep your response concise, authentic to your speaking style, and focused on the topic.
Use your own policies and perspectives to address the issue.
""",
        "should_deflect": False,
        "max_new_tokens": 1024,
        "max_length": 1536
    }
    
    # Generate the response
    try:
        response_data = generate_response(input_state)
        
        # Handle different response formats
        if isinstance(response_data, dict):
            # Extract just the response text from the response dictionary
            if 'response' in response_data:
                return response_data['response']
            else:
                # If no 'response' key is found, use the entire string representation as fallback
                return str(response_data)
        else:
            # Assume it's already a string
            return response_data
    except Exception as e:
        print(f"Error generating response for {politician_name}: {e}")
        # Provide a fallback response
        return f"As {politician_name}, I believe this is an important issue that requires thoughtful consideration. We need to work together to address {topic} with sensible policies that benefit all Americans." 