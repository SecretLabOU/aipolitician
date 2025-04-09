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
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
    import numpy as np
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False
    logging.warning("ChromaDB or sentence-transformers not installed. RAG functionality will be disabled.")

# Constants
DB_PATH = "/opt/chroma_db"
COLLECTION_NAME = "politicians"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_NUM_RESULTS = 5

# Initialize global variables
_client = None
_collection = None
_embedding_model = None

def init_chroma_db() -> Tuple[bool, Optional[str]]:
    """
    Initialize the ChromaDB client and collection.
    
    Returns:
        Tuple[bool, Optional[str]]: (Success status, Error message if any)
    """
    global _client, _collection, _embedding_model
    
    if not HAS_DEPENDENCIES:
        return False, "Required dependencies not installed"
    
    if not os.path.exists(DB_PATH):
        return False, f"Database path {DB_PATH} does not exist"
    
    try:
        # Initialize the ChromaDB client with persistent storage
        _client = chromadb.PersistentClient(
            path=DB_PATH,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=False
            )
        )
        
        # Get the politicians collection
        _collection = _client.get_collection(name=COLLECTION_NAME)
        
        # Initialize the embedding model
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        return True, None
        
    except Exception as e:
        logging.error(f"Error initializing ChromaDB: {str(e)}")
        return False, str(e)

def get_embeddings(text: str) -> List[float]:
    """
    Generate embeddings for a text using the SentenceTransformer model.
    
    Args:
        text (str): The text to generate embeddings for
        
    Returns:
        List[float]: The embedding vector
    """
    global _embedding_model
    
    if _embedding_model is None:
        success, error = init_chroma_db()
        if not success:
            logging.error(f"Failed to initialize embedding model: {error}")
            return []
    
    try:
        # Get embeddings and convert to list
        embedding = _embedding_model.encode(text)
        return embedding.tolist()
    except Exception as e:
        logging.error(f"Error generating embeddings: {str(e)}")
        return []

def query_database(
    query_text: str, 
    politician_name: str, 
    num_results: int = DEFAULT_NUM_RESULTS
) -> List[Dict[str, Any]]:
    """
    Query the ChromaDB database for relevant documents.
    
    Args:
        query_text (str): The query text
        politician_name (str): The name of the politician to filter by
        num_results (int): Maximum number of results to return
        
    Returns:
        List[Dict[str, Any]]: List of matching documents with metadata
    """
    global _collection
    
    if _collection is None:
        success, error = init_chroma_db()
        if not success:
            logging.error(f"Failed to initialize collection: {error}")
            return []
    
    try:
        # Generate embeddings for the query
        query_embedding = get_embeddings(query_text)
        
        # Search the collection with metadata filtering
        results = _collection.query(
            query_embeddings=[query_embedding],
            n_results=num_results,
            where={"politician_name": politician_name}
        )
        
        # Format the results
        documents = []
        for i, (doc, metadata) in enumerate(zip(results.get("documents", [[]])[0], results.get("metadatas", [[]])[0])):
            documents.append({
                "text": doc,
                "metadata": metadata,
                "score": results.get("distances", [[]])[0][i] if "distances" in results else None
            })
            
        return documents
        
    except Exception as e:
        logging.error(f"Error querying database: {str(e)}")
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
        # Query the database
        documents = query_database(query, politician_name)
        
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

# Optional initialization at module import time
if HAS_DEPENDENCIES:
    try:
        init_status, init_error = init_chroma_db()
        if not init_status:
            logging.warning(f"Failed to initialize ChromaDB: {init_error}")
    except Exception as e:
        logging.error(f"Unexpected error initializing ChromaDB: {str(e)}") 