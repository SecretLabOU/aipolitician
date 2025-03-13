#!/usr/bin/env python3
"""
Run script for AI Politician server.
This sets up the Python path correctly.
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import and run the server
from app.server import app
import uvicorn

if __name__ == "__main__":
    print("Starting AI Politician server...")
    uvicorn.run(app, host="localhost", port=8000)
