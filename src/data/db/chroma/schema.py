#!/usr/bin/env python3
"""
ChromaDB Schema and Connection Module for AI Politician RAG System

This module handles the connection to ChromaDB and defines the schema
for the politician document collection.
"""

import os
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

# Set the default database path
DEFAULT_DB_PATH = "/opt/chroma_db"
DEFAULT_COLLECTION_NAME = "politicians"

# Import ChromaDB
try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False
    logging.warning("ChromaDB not installed. RAG functionality will be disabled.")

def connect_to_chroma(db_path: str = DEFAULT_DB_PATH) -> Optional[Any]:
    """
    Connect to ChromaDB database.
    
    Args:
        db_path: Path to ChromaDB persistent storage
        
    Returns:
        ChromaDB client or None if connection fails
    """
    if not HAS_CHROMADB:
        return None
        
    if not os.path.exists(db_path):
        logging.error(f"Database path {db_path} does not exist")
        return None
        
    try:
        # Initialize the ChromaDB client with persistent storage
        client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=False
            )
        )
        
        # Verify connection by checking if the politicians collection exists
        try:
            collection = client.get_collection(name=DEFAULT_COLLECTION_NAME)
            if collection:
                logging.info(f"Successfully connected to ChromaDB at {db_path}")
                return client
        except Exception as e:
            logging.error(f"Collection '{DEFAULT_COLLECTION_NAME}' not found: {str(e)}")
            return None
            
    except Exception as e:
        logging.error(f"Error connecting to ChromaDB: {str(e)}")
        return None
        
    return None

def get_collection(client: Any, collection_name: str = DEFAULT_COLLECTION_NAME) -> Optional[Any]:
    """
    Get a collection from ChromaDB.
    
    Args:
        client: ChromaDB client
        collection_name: Name of the collection to retrieve
        
    Returns:
        ChromaDB collection or None if not found
    """
    if not client:
        return None
        
    try:
        collection = client.get_collection(name=collection_name)
        return collection
    except Exception as e:
        logging.error(f"Error getting collection '{collection_name}': {str(e)}")
        return None

def query_politician_data(
    collection: Any,
    query_text: str,
    politician_name: str,
    num_results: int = 5
) -> List[Dict[str, Any]]:
    """
    Query the politician collection for relevant documents.
    
    Args:
        collection: ChromaDB collection
        query_text: The text to search for
        politician_name: Name of the politician to filter by
        num_results: Maximum number of results to return
        
    Returns:
        List of document dictionaries with text and metadata
    """
    if not collection:
        return []
        
    try:
        # Import the embeddings utility to avoid circular imports
        from src.data.db.utils.rag_utils import get_embeddings
        
        # Generate embeddings for the query
        query_embedding = get_embeddings(query_text)
        
        if not query_embedding:
            return []
        
        # Search the collection with metadata filtering
        results = collection.query(
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