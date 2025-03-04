"""State management for the political agent graph.

This module defines the state structures used by the graph, including
input state and agent state for the graph's operation.
"""

from dataclasses import dataclass, field
from typing import Annotated, Dict, List, Optional, Any

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

@dataclass(kw_only=True)
class InputState:
    """Input state for the agent graph."""
    messages: Annotated[list[AnyMessage], add_messages]

@dataclass(kw_only=True)
class AgentState(InputState):
    """State for the political agent graph.
    
    Contains all the information needed to track conversation state,
    including persona details, sentiment analysis, context, and response generation.
    """
    # Persona information
    persona_id: str = ""
    persona_name: str = ""
    persona_details: Dict[str, Any] = field(default_factory=dict)
    persona_change_message: Optional[str] = None
    
    # Sentiment and context
    sentiment: str = ""
    emotions: Dict[str, float] = field(default_factory=dict)
    primary_emotion: str = ""
    emotional_context: str = ""
    context: str = ""
    
    # Routing and database selection
    selected_databases: List[str] = field(default_factory=list)
    recurring_topics: List[str] = field(default_factory=list)
    
    # Database results
    voting_data: str = ""
    bio_data: str = ""
    social_data: str = ""
    policy_data: str = ""
    aggregated_data: str = ""
    
    # Response generation
    tone: str = ""
    persona_style: str = ""
    deflection: str = ""
    draft_response: str = ""
    verified_response: str = ""
    
    # Context continuity
    conversation_context: str = ""
    
    # Analytics
    response_type: str = ""
