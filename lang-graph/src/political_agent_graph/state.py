"""State management for the political agent graph.

This module defines the state structures used in the political agent graph.
"""

from dataclasses import dataclass, field
from typing import Annotated, List

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


@dataclass(kw_only=True)
class InputState:
    """Input state for the agent."""
    messages: Annotated[list[AnyMessage], add_messages]


@dataclass(kw_only=True)
class AgentState(InputState):
    """State for the political agent graph."""
    # User input processing
    sentiment: str = ""
    context: str = ""
    
    # Routing and database selection
    selected_databases: list[str] = field(default_factory=list)
    
    # Database results
    voting_data: str = ""
    bio_data: str = ""
    social_data: str = ""
    policy_data: str = ""
    aggregated_data: str = ""
    
    # Response generation
    data_found: bool = True
    tone: str = ""
    persona_style: str = ""
    deflection: str = ""
    draft_response: str = ""
    verified_response: str = ""
    
    # Chat memory
    chat_memory: list[str] = field(default_factory=list)
