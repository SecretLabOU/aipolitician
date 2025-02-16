from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Optional
from app.models.chat import ChatRequest, ChatResponse, ErrorResponse
from app.agents.trump import TrumpAgent
from app.agents.biden import BidenAgent

router = APIRouter()

# Initialize agents
agents: Dict[str, any] = {
    "trump": TrumpAgent(),
    "biden": BidenAgent()
}

async def get_agent(agent_name: str):
    """Dependency to get agent by name."""
    agent_name = agent_name.lower()
    if agent_name not in agents:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_name}' not found. Available agents: {', '.join(agents.keys())}"
        )
    return agents[agent_name]

@router.post(
    "/chat/{agent_name}",
    response_model=ChatResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
)
async def chat_with_agent(
    agent_name: str,
    chat_request: ChatRequest,
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
        response = await agent.generate_response(
            message=chat_request.message,
            session_id=chat_request.session_id
        )
        return ChatResponse(**response)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {str(e)}"
        )

@router.get("/agents")
async def list_agents():
    """List available political agents."""
    return {
        "agents": list(agents.keys()),
        "total": len(agents)
    }
