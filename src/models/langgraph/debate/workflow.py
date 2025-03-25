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
    
    # Conditional routing from moderator
    workflow.add_conditional_edges(
        "moderator",
        lambda state: next_step(state),
        {
            "debater": "debater",
            "topic_manager": "topic_manager",
            "end": END
        }
    )
    
    # Politician debater speaking
    workflow.add_conditional_edges(
        "debater",
        lambda state: check_interruption(state),
        {
            "interruption": "interruption_handler",
            "fact_check": "fact_checker",
            "moderator": "moderator"
        }
    )
    
    # Fact checking results - add a check to force end after too many iterations
    workflow.add_conditional_edges(
        "fact_checker",
        lambda state: "end" if len(state.get("turn_history", [])) >= 15 else "moderator",
        {
            "moderator": "moderator",
            "end": END
        }
    )
    
    # Handle interruption - add a check to force end after too many iterations
    workflow.add_conditional_edges(
        "interruption_handler",
        lambda state: "end" if len(state.get("turn_history", [])) >= 15 else "moderator",
        {
            "moderator": "moderator",
            "end": END
        }
    )
    
    # Topic management - add a check to force end after too many iterations 
    workflow.add_conditional_edges(
        "topic_manager",
        lambda state: "end" if len(state.get("turn_history", [])) >= 15 else "moderator",
        {
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
        print(f"Format: {state['format']['format_type']}")
        print(f"Participants: {', '.join(state['participants'])}")
        print("---------------------------------")
    
    # Initialize the debate state
    result = state.copy()
    
    # Set up initial speaking queue based on format
    result["speaking_queue"] = state["participants"].copy()
    result["current_speaker"] = result["speaking_queue"].pop(0)
    result["current_subtopic"] = state["topic"]
    
    # Initialize states for each debater
    result["debater_states"] = {
        participant: {
            "identity": participant,
            "position": "",
            "knowledge": "",
            "pending_rebuttal": "",
            "interrupted": False
        } for participant in state["participants"]
    }
    
    # Initialize other tracking fields
    result["turn_history"] = []
    result["fact_checks"] = []
    result["moderator_notes"] = []
    result["interruption_requested"] = False
    
    return result


def next_step(state: DebateState) -> str:
    """Determine the next step in the debate workflow."""
    # Check if debate should end
    if len(state["turn_history"]) >= 10:  # End after 10 turns
        return "end"
    
    # Check if topic needs management
    if len(state["turn_history"]) % 3 == 0 and len(state["turn_history"]) > 0:
        return "topic_manager"
    
    # Default to next debater
    return "debater"


def check_interruption(state: DebateState) -> str:
    """Check if an interruption should occur."""
    if state["interruption_requested"] and state["format"]["allow_interruptions"]:
        return "interruption"
    
    # Check if fact checking is needed
    if state["format"]["fact_check_enabled"] and len(state["turn_history"]) > 0:
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
    
    # Convert to runnable and set a higher recursion limit
    # The error shows we need more than the default 25
    try:
        debate_chain = graph.compile({"recursion_limit": 100})
    except TypeError:
        # Fall back to older approach if needed
        debate_chain = graph.compile()
    
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
    
    # Run the workflow
    result = debate_chain.invoke(initial_state)
    
    return {
        "topic": result["topic"],
        "participants": result["participants"],
        "turn_history": result["turn_history"],
        "fact_checks": result["fact_checks"],
        "moderator_notes": result["moderator_notes"]
    } 