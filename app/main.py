from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import logging
import time
from app.sessions import SessionManager
from app.agents import get_agent
from app.models.secure_config import validate_model_path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Politician API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize session manager
session_manager = SessionManager(session_timeout_minutes=30)

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    response: str
    session_id: str

@app.on_event("startup")
def startup_event():
    """Validate model paths on startup"""
    if not validate_model_path("trump-mistral"):
        raise RuntimeError("Trump model files not found at specified path")

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/chat/{agent_name}")
async def chat(
    agent_name: str,
    request: ChatRequest
) -> dict:
    try:
        logger.info(f"Processing chat request for agent: {agent_name}")
        
        # Get or create session
        session = session_manager.get_session(request.session_id)
        
        # Initialize agent if needed
        if not session.agent:
            logger.info(f"Initializing new agent for session {request.session_id}")
            agent = get_agent(agent_name)
            if not agent:
                raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")
            session_manager.set_agent_for_session(request.session_id, agent)
        
        # Generate response
        logger.info("Generating response...")
        response = session.agent.generate_response(request.message, session.history)
        
        # Update history
        session.add_to_history(request.message, response)
        
        return {"response": response}
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/sessions/{session_id}")
def end_session(session_id: str):
    """End a chat session and clean up resources"""
    try:
        session_manager.cleanup_session(session_id)
        return {"status": "success", "message": "Session ended"}
    except Exception as e:
        logger.error(f"Error ending session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
