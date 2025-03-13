#!/usr/bin/env python3
"""
Run script for LangGraph Studio server.
This sets up the server for LangGraph Studio Web UI.
"""

import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import uvicorn
from src.langraph_studio import app

if __name__ == "__main__":
    print("Starting LangGraph Studio server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
