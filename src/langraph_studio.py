from langserve import add_routes
from fastapi import FastAPI
from political_agent_graph.graph import graph, ConversationState
from political_agent_graph.state import get_initial_state
from langchain.callbacks.tracers import LangChainTracer

# Create a FastAPI app
app = FastAPI()

# Add routes for the graph
add_routes(
    app,
    graph,
    path="/political-agent",
    input_type=ConversationState,
)

# For testing locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
