#!/usr/bin/env python3
"""
FastAPI server for the AI Politician LangGraph system.
This module exposes the LangGraph workflow as a REST API.
"""
import sys
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to the Python path
root_dir = Path(__file__).parent.parent.parent.parent.absolute()
sys.path.insert(0, str(root_dir))

# Load environment variables
load_dotenv()

from src.models.langgraph.workflow import process_user_input, PoliticianInput, PoliticianOutput

# Create FastAPI app
app = FastAPI(
    title="AI Politician API",
    description="API for interacting with AI politicians using LangGraph",
    version="1.0.0"
)

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "AI Politician API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.post("/api/politician/chat", response_model=PoliticianOutput)
async def chat(input_data: PoliticianInput):
    """
    Process a chat input through the AI Politician workflow.
    
    Args:
        input_data: The user input and configuration
        
    Returns:
        The politician's response and metadata
    """
    try:
        return process_user_input(input_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/politician/identities")
async def get_identities():
    """Get available politician identities."""
    return {
        "identities": ["biden", "trump"]
    }

def main():
    """Run the API server."""
    port = int(os.environ.get("AI_POLITICIAN_API_PORT", 8000))
    host = os.environ.get("AI_POLITICIAN_API_HOST", "127.0.0.1")
    
    print(f"Starting AI Politician API on http://{host}:{port}")
    print("API documentation available at http://localhost:8000/docs")
    
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main() 