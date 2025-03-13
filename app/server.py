from fastapi import FastAPI
from langserve import add_routes

# Import directly from the src directory
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.political_agent_graph.graph import graph, run_conversation
from src.political_agent_graph.state import ConversationState, get_initial_state

app = FastAPI(
    title="AI Politician",
    version="0.1.0",
    description="An AI politician simulation using LangGraph",
)

# Add the graph route with state
add_routes(
    app,
    graph,
    path="/political-agent",
    input_type=ConversationState,
)

# Add a simpler route that takes just user input
@app.post("/chat")
async def chat_endpoint(user_input: str):
    """Simple endpoint for chat."""
    response = await run_conversation(user_input)
    return {"response": response}

@app.post("/chat-with-trace")
async def chat_with_trace(user_input: str):
    """Chat with tracing enabled."""
    response = await run_conversation(user_input)
    return {"response": response, "trace_url": "https://smith.langchain.com/o/user/projects/ai-politician/traces"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)