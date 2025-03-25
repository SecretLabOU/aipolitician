"""
Conversation state management for political agent.

Defines the state structure and initialization for political conversations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class ConversationState:
    """
    Maintains the state of a conversation with a political agent.
    
    Attributes:
        user_input: Current user input text
        current_topic: Detected topic of conversation
        topic_sentiment: Detected sentiment toward the topic
        retrieved_context: Retrieved RAG context (if available)
        should_deflect: Whether to deflect from direct answer
        deflection_topic: Alternative topic for deflection 
        policy_stance: Generated policy stance
        fact_check_result: Result of fact checking
        final_response: Final formatted response
        conversation_history: Running history of the conversation
    """
    
    # Input
    user_input: str
    
    # Processing data
    current_topic: Optional[str] = None
    topic_sentiment: str = "neutral" 
    retrieved_context: str = ""
    
    # Strategy information
    should_deflect: bool = False
    deflection_topic: Optional[str] = None
    
    # Response generation
    policy_stance: Optional[str] = None
    fact_check_result: Optional[str] = None
    final_response: Optional[str] = None
    
    # Conversation history
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    
    def add_to_history(self, speaker: str, text: str) -> None:
        """Add a message to the conversation history."""
        self.conversation_history.append({
            "speaker": speaker,
            "text": text
        })
    
    def copy(self) -> 'ConversationState':
        """Create a copy of the current state."""
        return ConversationState(
            user_input=self.user_input,
            current_topic=self.current_topic,
            topic_sentiment=self.topic_sentiment,
            retrieved_context=self.retrieved_context,
            should_deflect=self.should_deflect,
            deflection_topic=self.deflection_topic,
            policy_stance=self.policy_stance,
            fact_check_result=self.fact_check_result,
            final_response=self.final_response,
            conversation_history=self.conversation_history.copy()
        )


def get_initial_state(user_input: str) -> ConversationState:
    """
    Create an initial state for a new user input.
    
    If this is a continuation of a conversation, preserves the history.
    
    Args:
        user_input: The user's message text
        
    Returns:
        A new ConversationState with the user input
    """
    # Add user input to history if there's existing history
    if hasattr(get_initial_state, "history"):
        history = getattr(get_initial_state, "history")
        history.append({"speaker": "User", "text": user_input})
    else:
        # First interaction, create new history
        history = [{"speaker": "User", "text": user_input}]
        
    # Store history for next time
    setattr(get_initial_state, "history", history)
    
    # Create new state with history
    return ConversationState(
        user_input=user_input,
        conversation_history=history.copy()
    )