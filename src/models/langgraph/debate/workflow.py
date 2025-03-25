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

# Add the project root to the Python path
root_dir = Path(__file__).parent.parent.parent.parent.parent.absolute()
sys.path.insert(0, str(root_dir))

from src.models.langgraph.config import PoliticianIdentity
from src.models.langgraph.debate.agents import (
    moderate_debate,
    politician_turn,
    fact_check,
    handle_interruption,
    manage_topic
)


# Define debate formats
class DebateFormat(BaseModel):
    """Configuration for the debate format."""
    format_type: Literal["town_hall", "head_to_head", "panel"] = Field(..., 
                                                         description="Type of debate format")
    time_per_turn: int = Field(default=60, description="Time in seconds allocated per turn")
    allow_interruptions: bool = Field(default=True, description="Whether interruptions are allowed")
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
        elif len(state.get("turn_history", [])) % 3 == 0 and len(state.get("turn_history", [])) > 0:
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
        print(f"Format: {state.get('format', {}).get('format_type', 'unknown')}")
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
            "format_type": "head_to_head",
            "time_per_turn": 60,
            "allow_interruptions": True,
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
        if len(state.get("turn_history", [])) % 3 == 0 and len(state.get("turn_history", [])) > 0:
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
    if state.get("interruption_requested", False) and state.get("format", {}).get("allow_interruptions", False):
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
    
    # Check LangGraph version to handle compatibility
    try:
        import langgraph
        import pkg_resources
        import logging
        
        # Configure logging for debugging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("debate_workflow")
        
        langgraph_version = pkg_resources.get_distribution("langgraph").version
        logger.info(f"Detected LangGraph version: {langgraph_version}")
        print(f"Detected LangGraph version: {langgraph_version}")
    except Exception as e:
        langgraph_version = "unknown"
        print(f"Could not detect LangGraph version: {e}")
    
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
    
    # Configure for LangGraph 0.3.8 specifically
    if langgraph_version == "0.3.8":
        print("Using LangGraph 0.3.8 specific configuration")
        
        try:
            # In LangGraph 0.3.8, we need to set debug mode
            debug_mode = True
            
            # In 0.3.8, recursion_limit is passed to the stream/invoke method, not compile
            debate_chain = graph.compile()
            
            # Inspect the graph
            print("Graph structure:")
            if hasattr(graph, "_graph"):
                print(f"Nodes: {list(graph._graph.nodes)}")
                print(f"Entry point: {graph._entry_point}")
                print(f"End node: {END}")
            
            # Get available config options
            print("Looking for available config options")
            try:
                from langgraph.prebuilt.config.constructor_config import _DEFAULT_CONFIG_KEYS
                print(f"Default config keys: {_DEFAULT_CONFIG_KEYS}")
            except ImportError:
                print("Could not import _DEFAULT_CONFIG_KEYS")
            
            # Try to extract available config options for specific graph
            try:
                config_keys = None
                if hasattr(debate_chain, "_config"):
                    config_keys = list(debate_chain._config.keys())
                elif hasattr(debate_chain, "config"):
                    config_keys = list(debate_chain.config.keys())
                print(f"Available config keys: {config_keys}")
            except Exception as e:
                print(f"Error getting config keys: {e}")
            
            # Run the workflow with debug flag and recursion limit 
            print("Starting debate workflow execution with extended recursion limit...")
            result = debate_chain.invoke(
                initial_state,
                {"debug": debug_mode, "recursion_limit": 100}
            )
            return {
                "topic": result["topic"],
                "participants": result["participants"],
                "turn_history": result["turn_history"],
                "fact_checks": result["fact_checks"],
                "moderator_notes": result["moderator_notes"]
            }
        except Exception as e:
            print(f"Error with LangGraph 0.3.8 approach: {e}")
            
            # Try alternative 0.3.8 approach - sometimes config is not passed via dict
            try:
                print("Trying alternative 0.3.8 approach with config as kwargs...")
                result = debate_chain.invoke(initial_state, debug=True, recursion_limit=100)
                return {
                    "topic": result["topic"],
                    "participants": result["participants"],
                    "turn_history": result["turn_history"],
                    "fact_checks": result["fact_checks"],
                    "moderator_notes": result["moderator_notes"]
                }
            except Exception as e:
                print(f"Alternative 0.3.8 approach failed: {e}")
    
    # Try multiple approaches for other versions
    try:
        # First attempt: Try using the LangGraph compile/invoke approach
        try:
            # Try the approach that works for latest LangGraph
            from langgraph.prebuilt import DispatchConfig
            config = DispatchConfig(recursion_limit=100)
            debate_chain = graph.compile(config=config)
        except ImportError:
            # DispatchConfig not available in this version
            try:
                # Try direct kwargs approach (works in some versions)
                debate_chain = graph.compile(recursion_limit=100, checkpointer=None)
            except Exception as e:
                print(f"Direct kwargs compilation failed: {e}")
                try:
                    # Try with NullCheckpointer
                    try:
                        from langgraph.checkpoint import NullCheckpointer
                        debate_chain = graph.compile(recursion_limit=100, checkpointer=NullCheckpointer())
                    except ImportError:
                        # If NullCheckpointer doesn't exist, just use basic compile
                        print("NullCheckpointer not available, using basic compile")
                        debate_chain = graph.compile()
                except Exception as e:
                    print(f"All attempted compilation methods failed: {e}")
                    print("Using simplest possible compile")
                    debate_chain = graph.compile()
        
        print("Starting debate workflow execution...")
        result = debate_chain.invoke(initial_state)
        
    except Exception as e:
        print(f"Error running debate with LangGraph: {e}")
        print("Falling back to manual debate simulation...")
        
        # Fallback: Run a simplified version of the debate manually
        # This completely bypasses the LangGraph StateGraph
        result = run_simplified_debate(initial_state)
    
    return {
        "topic": result["topic"],
        "participants": result["participants"],
        "turn_history": result["turn_history"],
        "fact_checks": result["fact_checks"],
        "moderator_notes": result["moderator_notes"]
    }


def run_simplified_debate(state: DebateState) -> DebateState:
    """
    Run a simplified debate without using LangGraph's StateGraph.
    This is a fallback method to use when there are compatibility issues.
    
    Args:
        state: Initial debate state
        
    Returns:
        Updated state after running the debate manually
    """
    print("Running simplified debate...")
    
    try:
        # Initialize the debate
        print("Initializing debate state...")
        result = initialize_debate(state)
        
        # Log initial state
        print(f"Initial participants: {result['participants']}")
        print(f"Initial speaker: {result['current_speaker']}")
        print(f"Topic: {result['topic']}")
        
        # Run a fixed number of turns (6 turns = 3 per participant)
        max_turns = 6
        current_turn = 0
        
        # Initialize with moderator
        print("Running moderator introduction...")
        result = moderate_debate(result)
        
        while current_turn < max_turns:
            try:
                # Log current turn information
                print(f"\nTurn {current_turn + 1}/{max_turns}")
                print(f"Current speaker: {result['current_speaker']}")
                
                # Politician's turn
                print(f"Running {result['current_speaker']}'s turn...")
                result = politician_turn(result)
                current_turn += 1
                
                # Check what happened in the turn
                latest_turn = result["turn_history"][-1] if result["turn_history"] else {}
                print(f"Speaker: {latest_turn.get('speaker', 'unknown')}")
                statement_excerpt = (latest_turn.get('statement', '')[:50] + '...') if latest_turn.get('statement') else 'No statement'
                print(f"Statement: {statement_excerpt}")
                
                # Fact checking
                if result["format"].get("fact_check_enabled", True):
                    print("Running fact checking...")
                    result = fact_check(result)
                
                # Handle interruptions
                if result.get("interruption_requested", False):
                    print(f"Handling interruption by {result.get('interrupt_by', 'unknown')}...")
                    result = handle_interruption(result)
                
                # Topic management (every 2 turns)
                if current_turn % 2 == 0 and current_turn > 0:
                    print("Managing debate topics...")
                    current_subtopic = result.get("current_subtopic", result["topic"])
                    print(f"Current subtopic: {current_subtopic}")
                    result = manage_topic(result)
                    
                    # Check if topic changed
                    new_subtopic = result.get("current_subtopic", result["topic"])
                    if new_subtopic != current_subtopic:
                        print(f"Topic changed to: {new_subtopic}")
                
                # Moderator transition
                print("Running moderator transition...")
                result = moderate_debate(result)
                
                # Check if we've reached max turns for this fallback approach
                if current_turn >= max_turns:
                    print(f"Reached maximum turns ({max_turns}), ending debate")
                    break
                    
            except Exception as e:
                print(f"Error during turn {current_turn}: {e}")
                # Try to continue with next turn if possible
                current_turn += 1
                if current_turn >= max_turns:
                    break
        
        # Add closing statement from moderator
        print("Adding closing statement...")
        result["moderator_notes"].append({
            "turn": len(result["turn_history"]),
            "message": f"This concludes our debate on {result['topic']}. Thank you to our participants for sharing their perspectives.",
            "timestamp": datetime.now().isoformat()
        })
        
        # Summary stats
        print("\nDebate summary:")
        print(f"Total turns: {len(result['turn_history'])}")
        print(f"Fact checks: {len(result['fact_checks'])}")
        print(f"Moderator notes: {len(result['moderator_notes'])}")
        
        return result
        
    except Exception as e:
        print(f"Critical error in simplified debate: {e}")
        # Return the original state with an error note if everything fails
        state["moderator_notes"] = state.get("moderator_notes", []) + [{
            "turn": 0,
            "message": f"The debate could not be completed due to an error: {str(e)}",
            "timestamp": datetime.now().isoformat(),
            "is_error": True
        }]
        return state 