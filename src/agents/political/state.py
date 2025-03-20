from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class PoliticalAgentState:
    """State for the political agent graph."""
    
    # Input
    query: str
    persona: str  # "trump" or "biden"
    
    # Analysis state
    sentiment: str = ""
    context: str = ""
    
    # RAG state
    retrieved_context: str = ""
    
    # Response state
    draft_response: str = ""
    final_response: str = ""
    
    # Chat history
    chat_memory: List[str] = field(default_factory=list)
