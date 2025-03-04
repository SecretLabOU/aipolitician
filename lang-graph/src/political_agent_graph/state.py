"""State management for the political agent graph.

This module defines the state schema for the LangGraph.
"""

from typing import Dict, List, Optional, Sequence, Any
from dataclasses import dataclass, field
import json


@dataclass
class ConversationState:
    """Conversation state for the political agent graph."""
    
    # User input
    user_input: str = ""
    
    # Conversation history
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    
    # Current topic information
    current_topic: Optional[str] = None
    topic_sentiment: Optional[str] = None
    
    # Strategy decisions
    should_deflect: bool = False
    deflection_topic: Optional[str] = None
    
    # Response components
    policy_stance: Optional[str] = None
    final_response: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for JSON serialization."""
        return {
            "user_input": self.user_input,
            "conversation_history": self.conversation_history,
            "current_topic": self.current_topic,
            "topic_sentiment": self.topic_sentiment,
            "should_deflect": self.should_deflect,
            "deflection_topic": self.deflection_topic,
            "policy_stance": self.policy_stance,
            "final_response": self.final_response,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationState":
        """Create state from dictionary."""
        return cls(
            user_input=data.get("user_input", ""),
            conversation_history=data.get("conversation_history", []),
            current_topic=data.get("current_topic"),
            topic_sentiment=data.get("topic_sentiment"),
            should_deflect=data.get("should_deflect", False),
            deflection_topic=data.get("deflection_topic"),
            policy_stance=data.get("policy_stance"),
            final_response=data.get("final_response"),
        )
    
    def add_to_history(self, speaker: str, text: str) -> None:
        """Add a message to the conversation history."""
        self.conversation_history.append({
            "speaker": speaker,
            "text": text,
        })


# Factory function for creating new conversation state
def get_initial_state(user_input: str) -> ConversationState:
    """Create an initial conversation state with user input."""
    state = ConversationState(user_input=user_input)
    state.add_to_history("User", user_input)
    return state