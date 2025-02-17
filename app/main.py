from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.sessions import SessionManager
from app.agents import get_agent

app = FastAPI()
session_manager = SessionManager()

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    session_id: str

@app.post("/chat/{agent_name}")
async def chat_with_agent(agent_name: str, request: ChatRequest) -> ChatResponse:
    agent = get_agent(agent_name)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    session = session_manager.get_session(request.session_id)
    response = agent.generate_response(request.message, session.history)
    session.add_interaction(request.message, response)
    
    return ChatResponse(response=response, session_id=request.session_id)