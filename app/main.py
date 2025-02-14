from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import torch

from app.models import load_model
from app.agents.base import BaseAgent

app = FastAPI(title="AI Politician Chat API")

# Initialize model and agent
print("Loading model...")
model, tokenizer = load_model()
agent = BaseAgent(model, tokenizer)

class ChatRequest(BaseModel):
    message: str
    agent_name: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with an AI politician.
    
    - message: The user's message
    - agent_name: Name of the politician (e.g., "trump" or "biden")
    - session_id: Optional session ID for continuing conversations
    """
    if request.agent_name.lower() not in ["trump", "biden"]:
        raise HTTPException(status_code=400, detail="Invalid agent name. Must be 'trump' or 'biden'")
    
    try:
        result = agent.chat(
            message=request.message,
            session_id=request.session_id,
            agent_name=request.agent_name.lower()
        )
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
