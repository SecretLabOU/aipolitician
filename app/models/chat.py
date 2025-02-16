from pydantic import BaseModel, Field
from typing import Optional, List, Dict

class ChatMessage(BaseModel):
    role: str = Field(..., description="The role of the message sender (user or assistant)")
    content: str = Field(..., description="The content of the message")

class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's message to the political agent")
    session_id: Optional[str] = Field(None, description="Session ID for continuing a conversation")

class ChatResponse(BaseModel):
    response: str = Field(..., description="The political agent's response")
    session_id: str = Field(..., description="Session ID for the conversation")
    sentiment: Dict[str, float] = Field(
        default_factory=dict,
        description="Sentiment analysis of the user's message"
    )
    context: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional context about the response"
    )

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    details: Optional[Dict[str, str]] = Field(None, description="Additional error details")
