#!/usr/bin/env python3
"""
Run script for AI Politician server.
This sets up the Python path correctly.
"""

import sys
import os
import argparse

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def run_app_server():
    # Import and run the main app server
    from app.server import app
    import uvicorn
    print("Starting AI Politician server...")
    uvicorn.run(app, host="localhost", port=8000)

def run_langgraph_studio():
    # Import and run the LangGraph Studio server
    from src.langraph_studio import app
    import uvicorn
    print("Starting LangGraph Studio server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AI Politician servers")
    parser.add_argument("--studio", action="store_true", help="Run LangGraph Studio server")
    args = parser.parse_args()
    
    if args.studio:
        run_langgraph_studio()
    else:
        run_app_server()
