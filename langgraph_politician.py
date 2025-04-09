#!/usr/bin/env python3
"""
AI Politician - LangGraph Implementation
=======================================

This is the main entry point for the LangGraph-based AI Politician system.
It simulates politicians like Biden and Trump using a LangGraph workflow.

Usage:
  python langgraph_politician.py chat --identity biden [--debug] [--trace] [--no-rag]
  python langgraph_politician.py process --identity biden --input "Your question here"
  python langgraph_politician.py visualize
  python langgraph_politician.py debate --help

For easier usage, see the helper scripts in the scripts/ directory.
"""
import sys
from pathlib import Path
import logging
import os

# Add the project root to the Python path
root_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(root_dir))

# Import the CLI running command directly, not the main function
from src.models.langgraph.cli import run_cli
from src.models.langgraph.debate.cli import main as run_debate_cli

# Set up basic logging
logging.basicConfig(level=logging.INFO)

def check_rag_availability():
    """Check if the RAG system is available and properly configured."""
    # Check for ChromaDB dependency
    try:
        import chromadb
    except ImportError:
        print("RAG database system not available: ChromaDB not installed.")
        print("To install: pip install -r requirements-rag.txt")
        return False
        
    # Check for sentence-transformers dependency
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("RAG database system not available: SentenceTransformer not installed.")
        print("To install: pip install -r requirements-rag.txt")
        return False
    
    # Check database directory
    db_path = "/opt/chroma_db"
    if not os.path.exists(db_path):
        print(f"RAG database system not available: Database path {db_path} does not exist.")
        print(f"To create: sudo mkdir -p {db_path} && sudo chown $USER:$USER {db_path}")
        return False
        
    # Try to initialize the database connection
    try:
        from src.data.db.chroma.schema import connect_to_chroma, DEFAULT_DB_PATH, get_collection
        
        # Connect to ChromaDB
        client = connect_to_chroma(db_path=DEFAULT_DB_PATH)
        if not client:
            print("RAG database system not available: Failed to connect to ChromaDB.")
            return False
            
        # Check if collection exists
        collection = get_collection(client)
        if not collection:
            print("RAG database system not available: Politicians collection not found in database.")
            return False
            
        # Check if the embeddings work
        try:
            from src.data.db.utils.rag_utils import get_embeddings
            test_embedding = get_embeddings("Test query")
            if not test_embedding:
                print("RAG database system not available: Failed to generate embeddings.")
                return False
        except Exception as e:
            print(f"RAG database system not available: Error testing embeddings: {str(e)}")
            return False
            
        print("RAG database system available and operational.")
        return True
            
    except Exception as e:
        print(f"RAG database system not available: {str(e)}")
        print("Running with synthetic responses.")
        return False

if __name__ == "__main__":
    # Check if we're running the debate system
    if len(sys.argv) > 1 and sys.argv[1] == "debate":
        # Remove the "debate" argument and pass the rest to the debate CLI
        sys.argv.pop(1)
        run_debate_cli()
    else:
        # Check if RAG is available, unless explicitly disabled
        if "--no-rag" not in sys.argv:
            rag_available = check_rag_availability()
            if not rag_available:
                print("Running with synthetic responses.")
        
        # Run the regular CLI
        run_cli() 