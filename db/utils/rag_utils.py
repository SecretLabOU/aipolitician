"""
RAG Utils - Functions for Retrieval-Augmented Generation with Milvus
This module provides utility functions to connect to Milvus vector database,
retrieve relevant context based on user queries, and format this information
for integration with the chat models.
"""

import os
import logging
import time
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from pymilvus import (
    connections,
    Collection,
    utility,
    MilvusException
)
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MILVUS_HOST = "localhost"
DEFAULT_MILVUS_PORT = "19530"
DEFAULT_COLLECTION_NAME = "documents"
DEFAULT_TOP_K = 5
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_DIM = 384  # Dimension of the embedding vector

def connect_to_milvus(
    host: str = None, 
    port: str = None,
    timeout: int = 10
) -> bool:
    """
    Establishes a connection to the Milvus server.
    
    Args:
        host: Milvus server host. If None, uses DEFAULT_MILVUS_HOST or env variable.
        port: Milvus server port. If None, uses DEFAULT_MILVUS_PORT or env variable.
        timeout: Connection timeout in seconds.
        
    Returns:
        bool: True if connection is successful, False otherwise.
    """
    # Get connection details from environment variables or use defaults
    host = host or os.getenv("MILVUS_HOST", DEFAULT_MILVUS_HOST)
    port = port or os.getenv("MILVUS_PORT", DEFAULT_MILVUS_PORT)
    
    logger.info(f"Connecting to Milvus at {host}:{port}")
    
    try:
        connections.connect(
            alias="default", 
            host=host, 
            port=port,
            timeout=timeout
        )
        # Verify the connection
        if utility.has_collection(DEFAULT_COLLECTION_NAME):
            logger.info(f"Connected to Milvus server. Collection '{DEFAULT_COLLECTION_NAME}' exists.")
        else:
            logger.warning(f"Connected to Milvus server, but collection '{DEFAULT_COLLECTION_NAME}' does not exist.")
        return True
    except MilvusException as e:
        logger.error(f"Failed to connect to Milvus: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error connecting to Milvus: {e}")
        return False

def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    """
    Loads the sentence embedding model.
    
    Args:
        model_name: Name of the SentenceTransformer model.
        
    Returns:
        SentenceTransformer: The loaded model.
    """
    try:
        logger.info(f"Loading embedding model: {model_name}")
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}")
        raise

def search_milvus(
    query_text: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: Optional[SentenceTransformer] = None,
    top_k: int = DEFAULT_TOP_K,
    search_params: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """
    Searches the Milvus collection for relevant documents.
    
    Args:
        query_text: The text query to search for.
        collection_name: Name of the Milvus collection.
        embedding_model: SentenceTransformer model for embedding the query.
        top_k: Number of results to return.
        search_params: Parameters for the search.
        
    Returns:
        List of dictionaries containing search results.
    """
    if not embedding_model:
        embedding_model = get_embedding_model()
    
    try:
        # Generate embedding for the query
        query_embedding = embedding_model.encode(query_text)
        
        # Get the collection
        collection = Collection(collection_name)
        collection.load()
        
        # Set default search parameters if not provided
        if not search_params:
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
        
        # Perform the search
        search_results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text", "metadata"]
        )
        
        # Format the results
        formatted_results = []
        for hits in search_results:
            for hit in hits:
                formatted_results.append({
                    "id": hit.id,
                    "score": hit.score,
                    "text": hit.entity.get("text"),
                    "metadata": hit.entity.get("metadata", {})
                })
        
        logger.info(f"Found {len(formatted_results)} relevant documents")
        return formatted_results
    
    except MilvusException as e:
        logger.error(f"Milvus search error: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error during search: {e}")
        return []
    finally:
        # Release collection resources
        try:
            collection.release()
        except:
            pass

def format_context_for_chat(search_results: List[Dict[str, Any]], max_token_length: int = 2000) -> str:
    """
    Formats the search results into a context string for the chat model.
    
    Args:
        search_results: The search results from Milvus.
        max_token_length: Maximum approximate token length for the context.
        
    Returns:
        Formatted context string.
    """
    if not search_results:
        return ""
    
    context_parts = []
    total_length = 0
    
    for i, result in enumerate(search_results):
        # Extract the text and format with a header
        text = result.get("text", "")
        if not text:
            continue
        
        # Rough token count estimation (1 token ≈ 4 characters)
        text_length = len(text) // 4
        
        # Stop adding if we exceed the maximum token length
        if total_length + text_length > max_token_length:
            break
        
        # Format the text with citation and score
        formatted_text = f"[Document {i+1} - Relevance: {result['score']:.2f}]\n{text}\n"
        context_parts.append(formatted_text)
        total_length += text_length
    
    return "\n".join(context_parts)

def integrate_with_chat(
    query: str,
    is_biden_model: bool = True,
    milvus_host: str = None,
    milvus_port: str = None,
    top_k: int = DEFAULT_TOP_K
) -> Tuple[str, bool]:
    """
    Main function to integrate RAG with the chat models.
    Connects to Milvus, searches for relevant context, and formats it.
    
    Args:
        query: The user's query.
        is_biden_model: True if Biden model, False if Trump model.
        milvus_host: Milvus host address.
        milvus_port: Milvus port.
        top_k: Number of results to retrieve.
        
    Returns:
        Tuple of (context_string, success_flag)
    """
    # Model-specific settings (can be customized based on model differences)
    model_name = "Biden" if is_biden_model else "Trump"
    logger.info(f"Integrating RAG for {model_name} model with query: {query}")
    
    # Connect to Milvus
    if not connect_to_milvus(host=milvus_host, port=milvus_port):
        logger.error("Failed to connect to Milvus, returning without context")
        return "", False
    
    try:
        # Get embedding model
        embedding_model = get_embedding_model()
        
        # Perform search
        search_results = search_milvus(
            query_text=query,
            embedding_model=embedding_model,
            top_k=top_k
        )
        
        # Format context
        context = format_context_for_chat(search_results)
        
        if context:
            logger.info(f"Successfully retrieved and formatted context ({len(context)} chars)")
            return context, True
        else:
            logger.warning("No relevant context found")
            return "", True
            
    except Exception as e:
        logger.error(f"Error in RAG integration: {e}")
        return "", False

def get_collection_stats(collection_name: str = DEFAULT_COLLECTION_NAME) -> Dict[str, Any]:
    """
    Get statistics about the collection.
    
    Args:
        collection_name: Name of the Milvus collection.
        
    Returns:
        Dictionary with collection statistics.
    """
    try:
        if not utility.has_collection(collection_name):
            return {"error": f"Collection {collection_name} does not exist"}
        
        collection = Collection(collection_name)
        stats = {
            "name": collection_name,
            "entity_count": collection.num_entities,
            "schema": collection.schema,
            "indexes": collection.index().params if collection.has_index() else None
        }
        return stats
    except Exception as e:
        logger.error(f"Error getting collection stats: {e}")
        return {"error": str(e)}

