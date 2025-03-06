import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
"""
AI Politician Milvus Search Utilities

This module provides functions for searching and retrieving political figure
information from the Milvus vector database.
"""

from pymilvus import Collection, connections, utility
from sentence_transformers import SentenceTransformer
import logging
import json
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize the sentence transformer model for generating embeddings
model = None

def get_embedding_model():
    """
    Get or initialize the sentence transformer model.
    
    Returns:
        SentenceTransformer: The embedding model
    """
    global model
    if model is None:
        try:
            # Using a well-supported model for embedding generation
            model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise
    return model

def connect_to_milvus(host="localhost", port="19530"):
    """
    Establish connection to Milvus server.
    
    Args:
        host (str): Milvus server hostname
        port (str): Milvus server port
        
    Returns:
        bool: True if connection was successful
    """
    try:
        connections.connect("default", host=host, port=port)
        logger.info(f"Connected to Milvus server at {host}:{port}")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to Milvus: {str(e)}")
        return False

def get_collection(collection_name="political_figures"):
    """
    Get a Milvus collection and ensure it's loaded.
    
    Args:
        collection_name (str): Name of the collection
        
    Returns:
        Collection: The Milvus collection
    """
    if not utility.has_collection(collection_name):
        raise ValueError(f"Collection '{collection_name}' does not exist")
    
    collection = Collection(name=collection_name)
    
    # Load collection if not already loaded
    if not collection.is_loaded():
        collection.load()
        logger.info(f"Collection '{collection_name}' loaded into memory")
    
    return collection

def search_political_figures(query, limit=5, output_fields=None):
    """
    Search for political figures based on semantic similarity to the query.
    
    Args:
        query (str): The search query
        limit (int): Maximum number of results to return
        output_fields (list): List of fields to include in the results
        
    Returns:
        list: List of search results
    """
    # Connect to Milvus
    if not connections.has_connection("default"):
        connect_to_milvus()
    
    # Default output fields if none provided
    if output_fields is None:
        output_fields = ["name", "political_affiliation", "biography"]
    
    # Get collection
    collection = get_collection()
    
    # Generate embedding for query
    model = get_embedding_model()
    query_embedding = model.encode(query).tolist()
    
    # Search parameters
    search_params = {
        "metric_type": "L2",
        "params": {"ef": 100}  # Size of final candidate list for search
    }
    
    # Execute search
    try:
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
                result = {field: hit.entity.get(field) for field in output_fields}
                result["score"] = 1.0 / (1.0 + hit.distance)  # Convert distance to similarity score
                search_results.append(result)
        
        return search_results
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise

def insert_political_figure(politician_data):
    """
    Insert a political figure into the database.
    
    Args:
        politician_data (dict): Dictionary containing politician information
        
    Returns:
        str: The ID of the inserted politician
    """
    # Connect to Milvus
    if not connections.has_connection("default"):
        connect_to_milvus()
    
    # Get collection
    collection = get_collection()
    
    # Ensure politician_data has all required fields
    required_fields = ["name", "biography"]
    for field in required_fields:
        if field not in politician_data:
            raise ValueError(f"Missing required field: {field}")
    
    # Generate embedding for biography
    model = get_embedding_model()
    embedding = model.encode(politician_data["biography"]).tolist()
    
    # Generate UUID if not provided
    if "id" not in politician_data:
        politician_data["id"] = str(uuid.uuid4())
    
    # Ad
