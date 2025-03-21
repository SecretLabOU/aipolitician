from dataclasses import dataclass, field
from typing import List

@dataclass
class PoliticalAgentState:
    """State for the political agent graph."""
    
    # Input
    query: str
    persona: str  # "trump" or "biden"
    
    # RAG state
    retrieved_context: str = ""
    
    # Response state
    final_response: str = ""
    
    # Optional: Chat history for context
    chat_memory: List[str] = field(default_factory=list)
