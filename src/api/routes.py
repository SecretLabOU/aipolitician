"""API routes for PoliticianAI."""

import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.agents import (
    DialogueGenerationAgent,
    WorkflowManager
)
from src.database import Session as DbSession
from src.database.models import ChatHistory, Politician, Statement, Topic
from src.utils import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Dependency to get database session
def get_db():
    """Get database session."""
    db = DbSession()
    try:
        yield db
    finally:
        db.close()

# Request/Response Models
class ChatRequest(BaseModel):
    """Chat request model."""
    
    message: str

class ChatResponse(BaseModel):
    """Chat response model."""
    
    response: str
    session_id: str

# Routes
@router.post("/chat/{agent}", response_model=ChatResponse)
async def chat(
    agent: str,
    request: ChatRequest,
    db: Session = Depends(get_db)
) -> ChatResponse:
    """Process chat message and return agent-specific response."""
    try:
        if agent not in ["trump", "biden"]:
            raise HTTPException(status_code=400, detail="Invalid agent specified")
            
        # Initialize workflow manager
        workflow = WorkflowManager()
        
        # Process message
        result = workflow.process_message(
            message=request.message,
            agent=agent,
            db=db
        )
        
        # Save to chat history
        history = ChatHistory(
            session_id=result["session_id"],
            user_input=request.message,
            system_response=result["response"]
        )
        db.add(history)
        db.commit()
        
        return ChatResponse(
            response=result["response"],
            session_id=result["session_id"]
        )
        
    except Exception as e:
        logger.error(f"Error processing chat message: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error processing chat message"
        )

@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}
