"""
RAG Utilities for AI Politician

This module provides Retrieval-Augmented Generation (RAG) utilities
for the AI Politician system, allowing it to retrieve relevant factual
information from the ChromaDB vector database.
"""
import sys
import os
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add the project root to the Python path
root_dir = Path(__file__).parent.parent.parent.parent.parent.absolute()
sys.path.insert(0, str(root_dir))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    from src.data.db.chroma.schema import connect_to_chroma, get_collection, DEFAULT_DB_PATH, DEFAULT_COLLECTION_NAME, BGEEmbeddingFunction
    HAS_DEPENDENCIES = True
except ImportError:
    logger.warning("RAG dependencies not found, running in fallback mode")
    HAS_DEPENDENCIES = False

# Global variables for caching
_embedding_model = None
_client = None
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
            # Using BGE-Small-EN for consistency with ChromaDB schema
            logger.info("Loading embedding model...")
            _embedding_model = SentenceTransformer('BAAI/bge-small-en')
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            return None
            
    return _embedding_model

def _connect_to_chromadb():
    """
    Connect to the ChromaDB server.
    
    Returns:
        bool: True if connection was successful, False otherwise
    """
    global _client
    
    if not HAS_DEPENDENCIES:
        return False
        
    try:
        # Connect to ChromaDB
        if _client is None:
            _client = connect_to_chroma(db_path=DEFAULT_DB_PATH)
            if _client:
                logger.info(f"Connected to ChromaDB at {DEFAULT_DB_PATH}")
                return True
            return False
        return True
    except Exception as e:
        logger.error(f"Failed to connect to ChromaDB: {str(e)}")
        return False

def _get_collection(collection_name=DEFAULT_COLLECTION_NAME):
    """
    Get a ChromaDB collection.
    
    Args:
        collection_name (str): Name of the collection
        
    Returns:
        Collection: The ChromaDB collection, or None if not available
    """
    global _collection_cache, _client
    
    if not HAS_DEPENDENCIES:
        return None
        
    # Check cache first
    if collection_name in _collection_cache:
        return _collection_cache[collection_name]
        
    # Connect to ChromaDB
    if not _connect_to_chromadb() or not _client:
        return None
        
    # Get collection
    try:
        collection = get_collection(_client, collection_name)
        if collection:
            # Cache the collection
            _collection_cache[collection_name] = collection
            return collection
        return None
    except Exception as e:
        logger.error(f"Error getting collection: {str(e)}")
        return None

def semantic_search(query, politician_name=None, limit=5):
    """
    Perform semantic search in the ChromaDB database.
    
    Args:
        query (str): The search query
        politician_name (str, optional): Name of the politician (e.g., 'biden', 'trump')
        limit (int): Maximum number of results to return
        
    Returns:
        List[Dict[str, Any]]: List of search results, or empty list if search failed
    """
    if not HAS_DEPENDENCIES:
        return []
    
    # Get collection
    collection = _get_collection()
    if collection is None:
        return []
        
    try:
        # Construct filters if politician specified
        where_filter = None
        if politician_name:
            # Convert politician name to match expected political_affiliation format
            politician_filter = politician_name.lower()
            
            # Map politician shorthand names to expected affiliation values
            affiliation_mapping = {
                "biden": "democrat",
                "trump": "republican"
            }
            
            # Use mapped value if available, otherwise use provided value
            affiliation = affiliation_mapping.get(politician_filter, politician_filter)
            
            # Filter by political_affiliation
            where_filter = {"political_affiliation": {"$eq": affiliation}}
        
        # Execute search
        results = collection.query(
            query_texts=[query],
            n_results=limit,
            where=where_filter,
            include=["metadatas", "documents", "distances"]
        )
        
        # Format results
        search_results = []
        if results and 'metadatas' in results and results['metadatas'] and results['metadatas'][0]:
            for i, metadata in enumerate(results['metadatas'][0]):
                result = {
                    "content": results.get('documents', [[]])[0][i] if results.get('documents') else "No content",
                    "politician": metadata.get('political_affiliation', "Unknown"),
                    "topic": "General",  # ChromaDB might not have this field
                    "source": metadata.get('source', "Unknown"),
                    "date": metadata.get('date', "Unknown"),
                    "score": float(results.get('distances', [[]])[0][i]) if results.get('distances') else 0.0
                }
                search_results.append(result)
        
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
    _connect_to_chromadb() 