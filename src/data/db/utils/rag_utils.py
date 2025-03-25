"""
RAG Utilities for AI Politician

This module provides Retrieval-Augmented Generation (RAG) utilities
for the AI Politician system, allowing it to retrieve relevant factual
information from the Milvus vector database.
"""
import sys
import os
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add the project root to the Python path
root_dir = Path(__file__).parent.parent.parent.parent.parent.absolute()
sys.path.insert(0, str(root_dir))

# Import database connection utilities
from src.data.db.milvus.connection import get_connection_params, get_collection_name

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from pymilvus import connections, Collection, utility
    from sentence_transformers import SentenceTransformer
    HAS_DEPENDENCIES = True
except ImportError:
    logger.warning("RAG dependencies not found, running in fallback mode")
    HAS_DEPENDENCIES = False

# Global variables for caching
_embedding_model = None
_collection_cache = {}

def _get_embedding_model():
    """
    Get or initialize the sentence transformer model.
    
    Returns:
        SentenceTransformer: The embedding model, or None if not available
    """
    global _embedding_model
    
    if not HAS_DEPENDENCIES:
        return None
        
    if _embedding_model is None:
        try:
            # Using a well-supported model for embedding generation
            logger.info("Loading embedding model...")
            _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            return None
            
    return _embedding_model

def _connect_to_milvus():
    """
    Connect to the Milvus server.
    
    Returns:
        bool: True if connection was successful, False otherwise
    """
    if not HAS_DEPENDENCIES:
        return False
        
    try:
        # Get connection parameters
        conn_params = get_connection_params()
        
        # Check if already connected
        if connections.has_connection(conn_params.get("alias", "default")):
            return True
            
        # Connect to Milvus
        connections.connect(**conn_params)
        logger.info(f"Connected to Milvus at {conn_params.get('host')}:{conn_params.get('port')}")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to Milvus: {str(e)}")
        return False

def _get_collection(collection_name):
    """
    Get a Milvus collection and ensure it's loaded.
    
    Args:
        collection_name (str): Name of the collection
        
    Returns:
        Collection: The Milvus collection, or None if not available
    """
    global _collection_cache
    
    if not HAS_DEPENDENCIES:
        return None
        
    # Check cache first
    if collection_name in _collection_cache:
        return _collection_cache[collection_name]
        
    # Connect to Milvus
    if not _connect_to_milvus():
        return None
        
    # Check if collection exists
    try:
        if not utility.has_collection(collection_name):
            logger.error(f"Collection '{collection_name}' does not exist")
            return None
            
        # Get collection
        collection = Collection(name=collection_name)
        
        # Load collection if not already loaded
        if not collection.is_loaded():
            collection.load()
            logger.info(f"Collection '{collection_name}' loaded into memory")
            
        # Cache the collection
        _collection_cache[collection_name] = collection
        return collection
    except Exception as e:
        logger.error(f"Error getting collection: {str(e)}")
        return None

def semantic_search(query, politician_name=None, limit=5):
    """
    Perform semantic search in the Milvus database.
    
    Args:
        query (str): The search query
        politician_name (str, optional): Name of the politician (e.g., 'biden', 'trump')
        limit (int): Maximum number of results to return
        
    Returns:
        List[Dict[str, Any]]: List of search results, or empty list if search failed
    """
    if not HAS_DEPENDENCIES:
        return []
        
    # Get embedding model
    model = _get_embedding_model()
    if model is None:
        return []
        
    # Get collection name based on politician
    collection_name = get_collection_name(politician_name)
    
    # Get collection
    collection = _get_collection(collection_name)
    if collection is None:
        return []
        
    try:
        # Generate embedding for query
        query_embedding = model.encode(query).tolist()
        
        # Define output fields
        output_fields = ["content", "politician", "topic", "source", "date"]
        
        # Search parameters for HNSW index
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 64}  # Higher ef gives better recall but slower search
        }
        
        # Execute search
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            output_fields=output_fields
        )
        
        # Format results
        search_results = []
        for hits in results:
            for hit in hits:
                result = {}
                for field in output_fields:
                    if field in hit.entity:
                        result[field] = hit.entity.get(field)
                        
                result["score"] = float(hit.score)  # Convert to float for JSON serialization
                search_results.append(result)
        
        # Filter results by politician if specified
        if politician_name and search_results:
            filtered_results = []
            for result in search_results:
                if result.get("politician") in [politician_name, "both"]:
                    filtered_results.append(result)
            
            # If filtering removed all results, return a few original results
            if not filtered_results and search_results:
                return search_results[:min(2, len(search_results))]
                
            return filtered_results
        
        return search_results
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        return []

def integrate_with_chat(prompt, politician_name=None):
    """
    Integrate RAG with the chat system by retrieving relevant information.
    
    Args:
        prompt (str): The user prompt
        politician_name (str, optional): Name of the politician (e.g., 'biden', 'trump')
        
    Returns:
        str: Retrieved information formatted for the chat system, or empty string if no results
    """
    # Search for relevant information
    results = semantic_search(prompt, politician_name=politician_name, limit=3)
    
    if not results:
        # Return empty string if no results
        return ""
        
    # Format the results for the chat system
    formatted_results = "Here is relevant information:\n\n"
    
    for i, result in enumerate(results, 1):
        content = result.get("content", "No content available")
        source = result.get("source", "Unknown source")
        date = result.get("date", "Unknown date")
        
        formatted_results += f"{i}. {content}\n"
        formatted_results += f"   Source: {source} ({date})\n\n"
    
    return formatted_results

# Initialize connection on module import
if HAS_DEPENDENCIES:
    _connect_to_milvus() 