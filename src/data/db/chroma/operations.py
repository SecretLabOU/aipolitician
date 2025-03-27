import os
import json
import logging
import uuid
from typing import Dict, List, Any, Optional, Union
import chromadb
from .schema import connect_to_chroma, get_collection, DEFAULT_DB_PATH, DEFAULT_COLLECTION_NAME

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def upsert_politician(collection: chromadb.Collection, 
                      politician_data: Dict[str, Any]) -> Optional[str]:
    """
    Insert or update a politician in the ChromaDB collection.
    
    Args:
        collection: ChromaDB collection
        politician_data: Dictionary containing politician data
        
    Returns:
        str: ID of the inserted document, or None if failed
    """
    try:
        # Generate or use existing ID
        doc_id = politician_data.get("id", str(uuid.uuid4()))
        
        # Create document for embedding
        # This is the text that will be used for semantic search
        texts_to_embed = [
            politician_data.get("name", ""),
            politician_data.get("biography", ""),
            politician_data.get("political_affiliation", "")
        ]
        
        # Add policy positions to the text for better semantic search
        policies = json.loads(politician_data.get("policies", "{}"))
        for policy_area, positions in policies.items():
            if isinstance(positions, list):
                for position in positions:
                    texts_to_embed.append(f"{policy_area}: {position}")
        
        # Concatenate all text
        document_text = " ".join([text for text in texts_to_embed if text])
        
        # Insert or update the document
        collection.upsert(
            ids=[doc_id],
            documents=[document_text],
            metadatas=[politician_data]
        )
        
        logger.info(f"Upserted politician {politician_data.get('name')} with ID {doc_id}")
        return doc_id
    except Exception as e:
        logger.error(f"Failed to upsert politician: {e}")
        return None

def search_politicians(collection: chromadb.Collection, 
                       query: str, 
                       n_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search for politicians in the ChromaDB collection.
    
    Args:
        collection: ChromaDB collection
        query: Search query
        n_results: Number of results to return
        
    Returns:
        List of politician data dictionaries
    """
    try:
        # Perform the search
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["metadatas", "documents", "distances"]
        )
        
        # Extract and format results
        politicians = []
        if results and 'metadatas' in results and results['metadatas']:
            for i, metadata in enumerate(results['metadatas'][0]):
                politician = metadata
                
                # Add distance score to the results
                if 'distances' in results and results['distances'] and len(results['distances'][0]) > i:
                    politician['distance'] = results['distances'][0][i]
                    
                politicians.append(politician)
                
        logger.info(f"Found {len(politicians)} politicians matching query: '{query}'")
        return politicians
    except Exception as e:
        logger.error(f"Failed to search politicians: {e}")
        return []

def get_politician_by_id(collection: chromadb.Collection, 
                         politician_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a politician by ID from the ChromaDB collection.
    
    Args:
        collection: ChromaDB collection
        politician_id: ID of the politician to retrieve
        
    Returns:
        Dict containing politician data, or None if not found
    """
    try:
        # Get the politician
        result = collection.get(
            ids=[politician_id],
            include=["metadatas", "documents"]
        )
        
        if result and 'metadatas' in result and result['metadatas']:
            logger.info(f"Retrieved politician with ID {politician_id}")
            return result['metadatas'][0]
        else:
            logger.warning(f"Politician with ID {politician_id} not found")
            return None
    except Exception as e:
        logger.error(f"Failed to get politician by ID: {e}")
        return None

def delete_politician(collection: chromadb.Collection, 
                      politician_id: str) -> bool:
    """
    Delete a politician from the ChromaDB collection.
    
    Args:
        collection: ChromaDB collection
        politician_id: ID of the politician to delete
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Delete the politician
        collection.delete(ids=[politician_id])
        logger.info(f"Deleted politician with ID {politician_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete politician with ID {politician_id}: {e}")
        return False

def get_all_politicians(collection: chromadb.Collection) -> List[Dict[str, Any]]:
    """
    Get all politicians from the ChromaDB collection.
    
    Args:
        collection: ChromaDB collection
        
    Returns:
        List of politician data dictionaries
    """
    try:
        # Get all politicians (limited to 10,000 for safety)
        result = collection.get(
            limit=10000,
            include=["metadatas"]
        )
        
        if result and 'metadatas' in result and result['metadatas']:
            logger.info(f"Retrieved {len(result['metadatas'])} politicians")
            return result['metadatas']
        else:
            logger.warning("No politicians found in collection")
            return []
    except Exception as e:
        logger.error(f"Failed to get all politicians: {e}")
        return [] 