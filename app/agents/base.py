from typing import Dict, List, Optional
from pydantic import BaseModel
from langgraph.graph import Graph
import uuid

class Message(BaseModel):
    role: str
    content: str

class ChatSession(BaseModel):
    session_id: str
    messages: List[Message]
    agent_name: str

class BaseAgent:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.sessions: Dict[str, ChatSession] = {}
        
    def _get_session(self, session_id: Optional[str] = None, agent_name: Optional[str] = None) -> ChatSession:
        """Get or create a chat session."""
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]
        
        new_session_id = session_id or str(uuid.uuid4())
        self.sessions[new_session_id] = ChatSession(
            session_id=new_session_id,
            messages=[],
            agent_name=agent_name or "default"
        )
        return self.sessions[new_session_id]

    def _build_prompt(self, messages: List[Message], agent_name: str) -> str:
        """Build the prompt for the model based on conversation history."""
        system_prompts = {
            "trump": "You are Donald Trump. Respond in his characteristic speaking style - direct, using simple words, often speaking in superlatives, and with his typical mannerisms. Keep responses concise and on-brand.",
            "biden": "You are Joe Biden. Respond in his characteristic speaking style - more formal, policy-focused, using traditional political speech patterns, and occasionally sharing personal anecdotes. Keep responses concise and on-brand."
        }
        
        prompt = f"{system_prompts.get(agent_name, 'You are a political figure.')}\n\n"
        
        for msg in messages[-5:]:  # Only use last 5 messages for context
            prompt += f"{msg.role}: {msg.content}\n"
        
        prompt += f"assistant: "
        return prompt

    def chat(self, message: str, session_id: Optional[str] = None, agent_name: str = "default") -> Dict:
        """Process a chat message and return a response."""
        from app.models import generate_response
        
        # Get or create session
        session = self._get_session(session_id, agent_name)
        
        # Add user message to history
        session.messages.append(Message(role="user", content=message))
        
        # Build prompt from conversation history
        prompt = self._build_prompt(session.messages, agent_name)
        
        # Generate response
        response = generate_response(self.model, self.tokenizer, prompt)
        
        # Add response to history
        session.messages.append(Message(role="assistant", content=response))
        
        return {
            "response": response,
            "session_id": session.session_id
        }
