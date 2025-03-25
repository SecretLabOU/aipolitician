#!/usr/bin/env python3
"""
AI Politician - LangGraph Implementation
=======================================

This is the main entry point for the LangGraph-based AI Politician system.
It simulates politicians like Biden and Trump using a LangGraph workflow.

Usage:
  python langgraph_politician.py cli chat --identity biden [--debug] [--trace] [--no-rag]
  python langgraph_politician.py cli process --identity biden --input "Your question here"
  python langgraph_politician.py cli visualize

For easier usage, see the helper scripts in the scripts/ directory.
"""
import sys
from pathlib import Path

# Add the project root to the Python path
root_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(root_dir))

from src.models.langgraph.cli import main

if __name__ == "__main__":
    # Check if Milvus database is available
    try:
        from pymilvus import connections
        from src.data.db.milvus.connection import get_connection_params
        
        # Try to connect to Milvus
        conn_params = get_connection_params()
        connections.connect(**conn_params)
        connections.disconnect(conn_params.get("alias", "default"))
    except Exception as e:
        print("RAG database system not available. Running with synthetic responses.")
    
    # Run the CLI
    main() 