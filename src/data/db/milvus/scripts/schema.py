import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

"""
AI Politician Milvus Database Schema

This module defines the schema for the Milvus database used to store and retrieve
political figure information for the AI debate system.
"""

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

def create_political_figures_collection(drop_existing=False):
    """
    Create the political_figures collection with appropriate schema.
    
    Args:
        drop_existing (bool): Whether to drop the collection if it exists
        
    Returns:
        Collection: The created or existing collection
    """
    collection_name = "political_figures"
    
    # Check if collection exists
    if utility.has_collection(collection_name):
        if drop_existing:
            logger.info(f"Dropping existing collection '{collection_name}'")
            utility.drop_collection(collection_name)
        else:
            logger.info(f"Collection '{collection_name}' already exists")
            return Collection(name=collection_name)
    
    # Define fields
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=36, is_primary=True),
        FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="date_of_birth", dtype=DataType.VARCHAR, max_length=10),  # Store as YYYY-MM-DD
        FieldSchema(name="nationality", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="political_affiliation", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="biography", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="positions", dtype=DataType.JSON),
        FieldSchema(name="policies", dtype=DataType.JSON),
        FieldSchema(name="legislative_actions", dtype=DataType.JSON),
        FieldSchema(name="public_communications", dtype=DataType.JSON),
        FieldSchema(name="timeline", dtype=DataType.JSON),
        FieldSchema(name="campaigns", dtype=DataType.JSON),
        FieldSchema(name="media", dtype=DataType.JSON),
        FieldSchema(name="philanthropy", dtype=DataType.JSON),
        FieldSchema(name="personal_details", dtype=DataType.JSON),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
    ]
    
    schema = CollectionSchema(fields, description="Political figures vector database for AI debate system")
    collection = Collection(name=collection_name, schema=schema)
    
    logger.info(f"Collection '{collection_name}' created successfully")
    return collection

def create_hnsw_index(collection_name="political_figures"):
    """
    Create HNSW index on the embedding field.
    
    Args:
        collection_name (str): Name of the collection to index
        
    Returns:
        Collection: The indexed collection
    """
    try:
        collection = Collection(name=collection_name)
        
        # Define index parameters
        index_params = {
            "index_type": "HNSW",
            "metric_type": "L2",
            "params": {
                "M": 16,                # Max number of connections per layer
                "efConstruction": 200,  # Size of dynamic candidate list
            }
        }
        
        # Create index on the embedding field
        collection.create_index(field_name="embedding", index_params=index_params)
        logger.info(f"HNSW index created on 'embedding' field in '{collection_name}' collection")
        
        # Configure search parameters
        collection.load()
        logger.info(f"Collection '{collection_name}' loaded into memory")
        
        return collection
    except Exception as e:
        logger.error(f"Failed to create index: {str(e)}")
        raise

def initialize_database():
    """
    Initialize the database by connecting, creating collection, and index.
    
    Returns:
        Collection: The initialized collection
    """
    if not connect_to_milvus():
        raise ConnectionError("Could not connect to Milvus database")
    
    collection = create_political_figures_collection()
    create_hnsw_index(collection.name)
    
    return collection

if __name__ == "__main__":
    print("Initializing Milvus database schema...")
    initialize_database()
    print("Schema initialization complete!")
