from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import logging
import time
import asyncio
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
async def startup_event():
    """Validate model paths on startup"""
    if not validate_model_path("trump-mistral"):
        raise RuntimeError("Trump model files not found at specified path")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/chat/{agent_name}")
async def chat_with_agent(
    agent_name: str,
    request: ChatRequest,
) -> ChatResponse:
    """Chat with an AI agent"""
    start_time = time.time()
    
    try:
        # Get or create session
        session = session_manager.get_session(request.session_id)
        
        # Initialize agent if needed
        if not session.agent:
            logger.info(f"Initializing new agent for session {request.session_id}")
            agent = get_agent(agent_name)
            if not agent:
                raise HTTPException(status_code=404, detail="Agent not found")
            session_manager.set_agent_for_session(request.session_id, agent)
        
        # Generate response with timeout
        try:
            logger.info("Starting response generation...")
            response = await session.agent.generate_response(request.message, session.history)
            
            # Validate response
            if not response or response.isspace():
                raise ValueError("Empty response generated")
            
            # Add to history
            session.add_interaction(request.message, response)
            
            # Log timing
            duration = time.time() - start_time
            logger.info(f"Response generated in {duration:.2f} seconds")
            
            return ChatResponse(response=response, session_id=request.session_id)
            
        except asyncio.TimeoutError:
            logger.error("Response generation timed out")
            raise HTTPException(
                status_code=504,
                detail="Response generation timed out. Please try again."
            )
        
    except Exception as e:
        logger.error(f"Error in chat_with_agent: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/sessions/{session_id}")
async def end_session(session_id: str):
    """End a chat session and clean up resources"""
    try:
        session_manager.cleanup_session(session_id)
        return {"status": "success", "message": "Session ended"}
    except Exception as e:
        logger.error(f"Error ending session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
