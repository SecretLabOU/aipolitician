"""API routes for PoliticianAI."""

import logging
from typing import Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.agents import WorkflowManager
from src.utils import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Request/Response Models
class ChatRequest(BaseModel):
    """Chat request model."""
    message: str

class ChatResponse(BaseModel):
    """Chat response model."""
    response: str

# Routes
@router.post("/chat/{agent}", response_model=ChatResponse)
async def chat(
    agent: str,
    request: ChatRequest
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
            agent=agent
        )
        
        return ChatResponse(
            response=result["response"]
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
