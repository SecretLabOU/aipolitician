from typing import Dict, Optional, List
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from transformers import pipeline
import uuid

class PoliticalAgent:
    def __init__(self, name: str, personality_traits: Dict[str, float]):
        """
        Initialize a political agent with a name and personality traits.
        
        Args:
            name: Name of the political figure
            personality_traits: Dictionary of personality traits and their strengths (0-1)
        """
        self.name = name
        self.personality_traits = personality_traits
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1  # CPU by default, will be updated if GPU is available
        )
        
        # Initialize conversation memory
        self.memories: Dict[str, ConversationBufferMemory] = {}
        
    def _create_session(self) -> str:
        """Create a new conversation session."""
        session_id = str(uuid.uuid4())
        self.memories[session_id] = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )
        return session_id
        
    def _get_memory(self, session_id: Optional[str] = None) -> ConversationBufferMemory:
        """Get or create memory for a session."""
        if not session_id or session_id not in self.memories:
            session_id = self._create_session()
        return self.memories[session_id], session_id
        
    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of input text."""
        result = self.sentiment_analyzer(text)[0]
        return {
            "label": result["label"],
            "score": result["score"]
        }
        
    def _format_response(self, response: str) -> str:
        """Format the response according to the agent's personality."""
        # To be implemented by specific agent classes
        return response
        
    async def generate_response(
        self,
        message: str,
        session_id: Optional[str] = None
    ) -> Dict:
        """
        Generate a response to the user's message.
        
        Args:
            message: User's input message
            session_id: Optional session ID for continuing a conversation
            
        Returns:
            Dictionary containing response, session_id, and sentiment analysis
        """
        # Get or create conversation memory
        memory, session_id = self._get_memory(session_id)
        
        # Analyze sentiment
        sentiment = self._analyze_sentiment(message)
        
        # Add user message to memory
        memory.chat_memory.add_message(HumanMessage(content=message))
        
        # Generate response (to be implemented by specific agent classes)
        response = "Base response - should be overridden"
        
        # Format response according to personality
        formatted_response = self._format_response(response)
        
        # Add agent response to memory
        memory.chat_memory.add_message(AIMessage(content=formatted_response))
        
        return {
            "response": formatted_response,
            "session_id": session_id,
            "sentiment": sentiment,
            "context": {
                "agent_name": self.name,
                "personality_traits": str(self.personality_traits)
            }
        }
        
    def set_gpu_device(self, device_id: int):
        """Set GPU device for the agent's models."""
        self.sentiment_analyzer.device = device_id
