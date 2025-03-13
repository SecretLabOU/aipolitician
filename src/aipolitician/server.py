from fastapi import FastAPI
from langchain.callbacks.tracers.langchain import wait_for_all_tracers
from langserve import add_routes
from political_agent_graph.graph import graph
from political_agent_graph.state import get_initial_state, ConversationState

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
    input_type=ConversationState,  # Use your state type
)

# Add a simpler route that just takes a string input
@app.post("/chat")
async def chat_endpoint(user_input: str):
    """Simple endpoint for chat that takes just the user input."""
    state = get_initial_state(user_input)
    result = await graph.ainvoke(state)
    await wait_for_all_tracers()
    return {"response": result.final_response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)