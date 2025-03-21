from dataclasses import dataclass, field
from typing import List, Dict, Any
from langchain_core.runnables import RunnableConfig

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
    
    def dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "query": self.query,
            "persona": self.persona,
            "retrieved_context": self.retrieved_context,
            "final_response": self.final_response,
            "chat_memory": self.chat_memory
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PoliticalAgentState":
        """Create state from dictionary."""
        return cls(
            query=data.get("query", ""),
            persona=data.get("persona", ""),
            retrieved_context=data.get("retrieved_context", ""),
            final_response=data.get("final_response", ""),
            chat_memory=data.get("chat_memory", [])
        )
