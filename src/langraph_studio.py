"""
LangGraph Studio integration for the AI Politician project.
This module sets up the LangGraph Studio server for visualizing and debugging the graph.
"""

from fastapi import FastAPI
from langserve import add_routes

# Import the graph and state from the political agent graph
from src.political_agent_graph.graph import graph
from src.political_agent_graph.state import ConversationState

# Create a FastAPI app
app = FastAPI(
    title="AI Politician LangGraph",
    version="0.1.0",
    description="LangGraph for simulating political conversations"
)

# Add routes for the graph
add_routes(
    app,
    graph,
    path="/political-agent",
    input_type=ConversationState,
)

# Add health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# For running the server directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
