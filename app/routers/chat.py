from fastapi import APIRouter, HTTPException, Depends, Request, Response
from typing import Dict, Optional
from app.models.chat import ChatRequest, ChatResponse, ErrorResponse
from app.agents.trump import TrumpAgent
from app.agents.biden import BidenAgent
from app.utils.rate_limiter import check_rate_limit
from app.utils.exceptions import (
    ModelLoadError,
    ModelGenerationError,
    SessionError,
    AgentNotFoundError,
    InvalidRequestError,
    handle_chat_error
)

router = APIRouter()

# Initialize agents with error handling
agents: Dict[str, any] = {}
try:
    agents = {
        "trump": TrumpAgent(),
        "biden": BidenAgent()
    }
except Exception as e:
    print(f"Error initializing agents: {str(e)}")

async def get_agent(agent_name: str):
    """Dependency to get agent by name."""
    agent_name = agent_name.lower()
    if not agents:
        raise ModelLoadError("Political agents are not initialized")
    if agent_name not in agents:
        raise AgentNotFoundError(
            f"Agent '{agent_name}' not found. Available agents: {', '.join(agents.keys())}"
        )
    return agents[agent_name]

@router.post(
    "/chat/{agent_name}",
    response_model=ChatResponse,
    responses={
        404: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse}
    }
)
async def chat_with_agent(
    request: Request,
    response: Response,
    agent_name: str,
    chat_request: ChatRequest,
    rate_limit_headers: Dict = Depends(check_rate_limit),
    agent: any = Depends(get_agent)
):
    """
    Chat with a political agent.
    
    Args:
        agent_name: Name of the political agent (trump/biden)
        chat_request: User message and optional session ID
        
    Returns:
        ChatResponse containing the agent's response and session information
    """
    try:
        # Add rate limit headers to response
        for header, value in rate_limit_headers.items():
            response.headers[header] = value
        
        # Validate request
        if not chat_request.message.strip():
            raise InvalidRequestError("Message cannot be empty")
        
        # Generate response
        agent_response = await agent.generate_response(
            message=chat_request.message,
            session_id=chat_request.session_id
        )
        
        return ChatResponse(**agent_response)
        
    except Exception as e:
        error_response = handle_chat_error(e)
        raise HTTPException(
            status_code=error_response["status_code"],
            detail=error_response["detail"]
        )

@router.get("/agents")
async def list_agents():
    """List available political agents."""
    if not agents:
        raise ModelLoadError("Political agents are not initialized")
    return {
        "agents": list(agents.keys()),
        "total": len(agents),
        "status": "healthy"
    }
