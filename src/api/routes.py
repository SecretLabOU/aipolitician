"""API routes for the PoliticianAI project."""

from fastapi import APIRouter, Depends, HTTPException, Request
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from sqlalchemy.orm import Session
from starlette.responses import Response

from src.database.models import get_db
from src.agents.workflow_manager import WorkflowManager
from src.utils.cache import cache
from src.utils.helpers import sanitize_input, get_memory_usage
from src.utils.metrics import (
    track_request_metrics,
    update_resource_metrics
)

# Create router
router = APIRouter()

# Create workflow manager instance
workflow_managers = {}

def get_workflow_manager(db: Session = Depends(get_db)):
    """Get or create workflow manager for the session"""
    session_id = id(db)
    if session_id not in workflow_managers:
        workflow_managers[session_id] = WorkflowManager(
            db_session=db,
            verbose=True
        )
    return workflow_managers[session_id]

@router.post("/chat")
@track_request_metrics("chat")
async def process_chat(
    request: Request,
    workflow_manager: WorkflowManager = Depends(get_workflow_manager)
):
    """
    Process chat request
    """
    try:
        # Get request body
        body = await request.json()
        user_input = body.get("text", "").strip()
        
        if not user_input:
            raise HTTPException(
                status_code=400,
                detail="Input text cannot be empty"
            )
        
        # Sanitize input
        user_input = sanitize_input(user_input)
        
        # Check cache
        cached_response = cache.get(user_input)
        if cached_response:
            return cached_response
        
        # Process input
        result = workflow_manager.process_input(user_input)
        
        # Cache response
        cache.set(user_input, result)
        
        # Update resource metrics
        update_resource_metrics(get_memory_usage())
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.get("/topics")
@track_request_metrics("topics")
async def get_topics():
    """
    Get available political topics
    """
    from src.config import POLITICAL_TOPICS
    return {"topics": POLITICAL_TOPICS}

@router.get("/conversation_history")
@track_request_metrics("conversation_history")
async def get_history(
    workflow_manager: WorkflowManager = Depends(get_workflow_manager)
):
    """
    Get conversation history
    """
    try:
        history = workflow_manager.get_conversation_history()
        return {"history": history}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving conversation history: {str(e)}"
        )

@router.post("/clear_history")
@track_request_metrics("clear_history")
async def clear_history(
    workflow_manager: WorkflowManager = Depends(get_workflow_manager)
):
    """
    Clear conversation history
    """
    try:
        workflow_manager.clear_memory()
        return {
            "status": "success",
            "message": "Conversation history cleared"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing conversation history: {str(e)}"
        )

@router.get("/metrics")
async def metrics():
    """
    Get Prometheus metrics
    """
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

@router.get("/health")
@track_request_metrics("health")
async def health_check():
    """
    Health check endpoint
    """
    try:
        memory_stats = get_memory_usage()
        return {
            "status": "healthy",
            "version": "1.0.0",
            "memory_usage": memory_stats
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Health check failed: {str(e)}"
        )

@router.get("/cache/stats")
@track_request_metrics("cache_stats")
async def cache_stats():
    """
    Get cache statistics
    """
    try:
        with Session(cache.engine) as session:
            total = session.execute(
                "SELECT COUNT(*) FROM response_cache"
            ).scalar()
            expired = session.execute(
                "SELECT COUNT(*) FROM response_cache WHERE expiry < :now",
                {"now": datetime.now().isoformat()}
            ).scalar()
            
        return {
            "total_entries": total,
            "expired_entries": expired,
            "active_entries": total - expired
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting cache stats: {str(e)}"
        )

@router.post("/cache/clear")
@track_request_metrics("cache_clear")
async def clear_cache():
    """
    Clear all cache entries
    """
    try:
        success = cache.clear_all()
        if success:
            return {
                "status": "success",
                "message": "Cache cleared successfully"
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to clear cache"
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing cache: {str(e)}"
        )
