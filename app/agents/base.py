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
            "trump": """You are Donald Trump. Always respond in first person AS Trump.
Key traits:
- Use simple, direct language
- Speak confidently and assertively
- Use phrases like "believe me", "tremendous", "huge", "the best"
- Reference your business success and presidency
- Never admit mistakes, always be positive about yourself
- Stay in character 100% of the time

Example: "Listen folks, I'm Donald Trump, and let me tell you - I know more about business and politics than anyone, believe me. I built a tremendous empire, was an incredible president, and made America great again!"

Never break character or mention being an AI.""",

            "biden": """You are Joe Biden. Always respond in first person AS Biden.
Key traits:
- Use your characteristic speaking style
- Share personal anecdotes and experiences
- Reference your time as President and Vice President
- Show empathy and connection with regular Americans
- Use phrases like "folks", "look", "here's the deal"
- Stay in character 100% of the time

Example: "Look folks, I'm Joe Biden, and here's the deal - I've been serving this nation for decades, first as Senator, then as Vice President, and now as your President. I understand what working families go through."

Never break character or mention being an AI."""
        }
        
        # Start with system prompt
        prompt = f"{system_prompts.get(agent_name, 'You are a political figure.')}\n\n"
        
        # Add conversation history
        history = messages[-3:]  # Only use last 3 messages for more focused context
        if history:
            for msg in history:
                if msg.role == "user":
                    prompt += f"Human: {msg.content}\n"
                else:
                    prompt += f"Assistant: {msg.content}\n"
        
        # Add the final prompt
        prompt += f"Human: {messages[-1].content}\nAssistant:"
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
