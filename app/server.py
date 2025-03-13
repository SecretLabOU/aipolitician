from fastapi import FastAPI
from langserve import add_routes
from src.political_agent_graph.graph import graph
from src.political_agent_graph.state import get_initial_state, ConversationState
from src.political_agent_graph import run_conversation_with_tracing

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
    state = get_initial_state(user_input)
    result = await graph.ainvoke(state)
    return {"response": result.final_response}

@app.post("/chat-with-trace")
async def chat_with_trace(user_input: str):
    """Chat with tracing enabled."""
    response = await run_conversation_with_tracing(user_input)
    return {"response": response, "trace_url": "https://smith.langchain.com/o/user/projects/ai-politician/traces"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)