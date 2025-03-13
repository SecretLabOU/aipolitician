"""
LangGraph Studio Web integration for AI Politician.

This module exposes the AI Politician graph to the LangGraph Studio Web UI.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph_api import RouteWebhook
from langgraph_api.base import LangGraphAPI

from src.political_agent_graph.graph import graph
from src.political_agent_graph.state import ConversationState

# Create a FastAPI app
app = FastAPI(
    title="AI Politician LangGraph Studio",
    version="0.1.0",
    description="LangGraph Studio Web integration for AI Politician",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a LangGraphAPI instance
api = LangGraphAPI(app)

# Register the graph
api.add_graph(
    name="ai_politician",
    graph=graph,
    input_type=ConversationState,
)

# Add a webhook route for the graph
webhook = RouteWebhook(graph=graph)
app.post("/webhook/ai_politician")(webhook)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
