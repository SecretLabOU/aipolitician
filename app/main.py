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
    try:
        agent = get_agent(agent_name)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        session = session_manager.get_session(request.session_id)
        response = agent.generate_response(request.message, session.history)
        
        if not response or response.isspace():
            raise ValueError("Empty response generated")
            
        session.add_interaction(request.message, response)
        return ChatResponse(response=response, session_id=request.session_id)
        
    except Exception as e:
        print(f"Error in chat_with_agent: {str(e)}")  # Debug logging
        raise HTTPException(status_code=500, detail=str(e))
