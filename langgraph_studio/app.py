"""
LangGraph Studio Web integration for AI Politician.

This module exposes the AI Politician graph to the LangGraph Studio Web UI.
"""

from fastapi import FastAPI
from langgraph.api import RouteWebhook
from langgraph.api.base import LangGraphAPI

from src.political_agent_graph.graph import graph, get_initial_state
from src.political_agent_graph.state import ConversationState

# Create a FastAPI app
app = FastAPI(
    title="AI Politician LangGraph Studio",
    version="0.1.0",
    description="LangGraph Studio Web integration for AI Politician",
)

# Create a LangGraphAPI instance
api = LangGraphAPI(app)

# Register the graph
api.add_graph(
    name="ai_politician",
    graph=graph,
    input_type=ConversationState,
    config_keys=["user_input"],
)

# Add a webhook route for the graph
webhook = RouteWebhook(graph=graph)
app.post("/webhook/ai_politician")(webhook)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)