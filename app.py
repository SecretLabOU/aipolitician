"""
LangGraph Studio Web integration for AI Politician.

This module exposes the AI Politician graph to the LangGraph Studio Web UI.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

# Create a simple endpoint to run the graph
@app.post("/run")
async def run_graph(input_data: dict):
    # Create initial state
    state = ConversationState(user_input=input_data.get("user_input", ""))
    
    # Run the graph
    result = await graph.ainvoke(state)
    
    # Return the result
    return {"response": result.final_response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
