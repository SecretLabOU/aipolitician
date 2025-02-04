"""API routes for PoliticianAI."""

import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.agents import (
    ContextAgent,
    ResponseAgent,
    SentimentAgent,
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
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    """Chat response model."""
    
    response: str
    sentiment: float
    topics: List[str]
    session_id: str

class TopicResponse(BaseModel):
    """Topic response model."""
    
    id: int
    name: str
    description: Optional[str] = None

class PoliticianResponse(BaseModel):
    """Politician response model."""
    
    id: int
    name: str
    party: Optional[str] = None
    position: Optional[str] = None
    bio: Optional[str] = None

# Routes
@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db)
) -> ChatResponse:
    """Process chat message and return response."""
    try:
        # Initialize workflow manager
        workflow = WorkflowManager()
        
        # Process message
        result = workflow.process_message(
            message=request.message,
            session_id=request.session_id,
            db=db
        )
        
        # Save to chat history
        history = ChatHistory(
            session_id=result["session_id"],
            user_input=request.message,
            system_response=result["response"],
            sentiment_score=result["sentiment"],
            context_topics=",".join(map(str, result["topic_ids"]))
        )
        db.add(history)
        db.commit()
        
        return ChatResponse(
            response=result["response"],
            sentiment=result["sentiment"],
            topics=result["topics"],
            session_id=result["session_id"]
        )
        
    except Exception as e:
        logger.error(f"Error processing chat message: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error processing chat message"
        )

@router.get("/topics", response_model=List[TopicResponse])
async def get_topics(
    db: Session = Depends(get_db)
) -> List[TopicResponse]:
    """Get list of available topics."""
    try:
        topics = db.query(Topic).all()
        return [
            TopicResponse(
                id=topic.id,
                name=topic.name,
                description=topic.description
            )
            for topic in topics
        ]
        
    except Exception as e:
        logger.error(f"Error fetching topics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error fetching topics"
        )

@router.get("/politicians", response_model=List[PoliticianResponse])
async def get_politicians(
    db: Session = Depends(get_db)
) -> List[PoliticianResponse]:
    """Get list of politicians."""
    try:
        politicians = db.query(Politician).all()
        return [
            PoliticianResponse(
                id=politician.id,
                name=politician.name,
                party=politician.party,
                position=politician.position,
                bio=politician.bio
            )
            for politician in politicians
        ]
        
    except Exception as e:
        logger.error(f"Error fetching politicians: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error fetching politicians"
        )

@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}
