#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) Utilities for AI Politician

This module provides functions to interact with a ChromaDB database for retrieving 
context about politicians to enhance AI responses with factual information.

The database stores documents about politicians with embeddings created using
the sentence-transformers/all-MiniLM-L6-v2 model.

Database Location: /opt/chroma_db (persistent storage)
Collection Name: politicians
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Import ChromaDB and SentenceTransformer libraries
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    # We'll import chromadb through the schema module to avoid duplication
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False
    logging.warning("Required dependencies not installed. RAG functionality will be disabled.")

# Constants
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize global variables
_embedding_model = None

def get_embedding_model() -> Optional[Any]:
    """
    Get or initialize the embedding model.
    
    Returns:
        SentenceTransformer model or None if initialization fails
    """
    global _embedding_model
    
    if _embedding_model is not None:
        return _embedding_model
        
    if not HAS_DEPENDENCIES:
        return None
    
    try:
        # Initialize the embedding model
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return _embedding_model
    except Exception as e:
        logging.error(f"Error initializing embedding model: {str(e)}")
        return None

def get_embeddings(text: str) -> List[float]:
    """
    Generate embeddings for a text using the SentenceTransformer model.
    
    Args:
        text (str): The text to generate embeddings for
        
    Returns:
        List[float]: The embedding vector
    """
    model = get_embedding_model()
    
    if model is None:
        logging.error("Failed to initialize embedding model")
        return []
    
    try:
        # Get embeddings and convert to list
        embedding = model.encode(text)
        return embedding.tolist()
    except Exception as e:
        logging.error(f"Error generating embeddings: {str(e)}")
        return []

def integrate_with_chat(query: str, politician_name: str) -> str:
    """
    Integrate RAG with chat by retrieving relevant context from the database.
    This function is called by the chat modules to enhance responses.
    
    Args:
        query (str): The user's query
        politician_name (str): The name of the politician (e.g., "Joe Biden", "Donald Trump")
        
    Returns:
        str: Formatted context to use in the prompt
    """
    if not HAS_DEPENDENCIES:
        return ""
    
    try:
        # Import here to avoid circular imports
        from src.data.db.chroma.schema import connect_to_chroma, get_collection, query_politician_data
        
        # Connect to ChromaDB
        client = connect_to_chroma()
        if not client:
            logging.warning("Failed to connect to ChromaDB")
            return ""
            
        # Get the politicians collection
        collection = get_collection(client)
        if not collection:
            logging.warning("Failed to get collection from ChromaDB")
            return ""
        
        # Query the database
        documents = query_politician_data(collection, query, politician_name)
        
        if not documents:
            return ""
        
        # Format the context
        context = "Here is some relevant factual information to help with your response:\n\n"
        
        for i, doc in enumerate(documents, 1):
            source = doc["metadata"].get("source", "Unknown source")
            content_type = doc["metadata"].get("content_type", "")
            
            context += f"{i}. {doc['text']}\n"
            if source:
                context += f"   Source: {source}\n"
            if content_type:
                context += f"   Type: {content_type}\n"
            context += "\n"
        
        return context
        
    except Exception as e:
        logging.error(f"Error integrating with chat: {str(e)}")
        return "" 