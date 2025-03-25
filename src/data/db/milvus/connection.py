"""
Milvus Connection Utilities

This module provides connection utilities for the Milvus database.
"""
import os
from typing import Dict, Any

# Default connection parameters - these can be overridden by environment variables
DEFAULT_HOST = "localhost"
DEFAULT_PORT = "19530"
DEFAULT_ALIAS = "default"

def get_connection_params() -> Dict[str, Any]:
    """
    Get connection parameters for Milvus, using environment variables if available.
    
    Returns:
        Dict[str, Any]: Dictionary with connection parameters
    """
    return {
        "host": os.getenv("MILVUS_HOST", DEFAULT_HOST),
        "port": os.getenv("MILVUS_PORT", DEFAULT_PORT),
        "alias": os.getenv("MILVUS_ALIAS", DEFAULT_ALIAS)
    }

def get_collection_name(politician_name: str = None) -> str:
    """
    Get the collection name for RAG, with optional filtering by politician.
    
    Args:
        politician_name (str, optional): Name of the politician (e.g., 'biden', 'trump')
        
    Returns:
        str: Collection name to use
    """
    # Default collection name for political knowledge
    collection_name = os.getenv("MILVUS_COLLECTION", "political_knowledge")
    
    # If a politician name is provided, we might want to use a politician-specific collection
    if politician_name and os.getenv(f"MILVUS_COLLECTION_{politician_name.upper()}"):
        return os.getenv(f"MILVUS_COLLECTION_{politician_name.upper()}")
    
    return collection_name 