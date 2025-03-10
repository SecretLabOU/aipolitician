#!/usr/bin/env python3
"""
Verify the state of the 'political_figures' collection.
"""
import os
import sys
import logging
from pymilvus import connections, Collection, utility

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def connect_to_milvus(host="localhost", port="19530"):
    """Establish connection to Milvus server."""
    try:
        connections.connect("default", host=host, port=port)
        logger.info(f"Connected to Milvus server at {host}:{port}")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to Milvus: {str(e)}")
        return False

def verify_collection(collection_name="political_figures"):
    """
    Verify the state of a collection.
    """
    if not utility.has_collection(collection_name):
        logger.error(f"Collection '{collection_name}' does not exist")
        return False
        
    try:
        collection = Collection(name=collection_name)
        
        # Load the collection before operating on it
        try:
            collection.load()
            logger.info(f"Collection '{collection_name}' loaded into memory")
        except Exception as e:
            logger.warning(f"Could not explicitly load collection: {str(e)}")
        
        # Get the number of entities
        entity_count = collection.num_entities
        logger.info(f"Collection '{collection_name}' has {entity_count} entities")
        
        # Print schema details
        schema = collection.schema
        logger.info(f"Collection schema: {schema}")
        
        field_names = [field.name for field in schema.fields]
        logger.info(f"Fields: {field_names}")
        
        # Check for index
        index_info = collection.index().params
        logger.info(f"Index info: {index_info}")
        
        return True
            
    except Exception as e:
        logger.error(f"Error verifying collection: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function."""
    if not connect_to_milvus():
        logger.error("Failed to connect to Milvus server")
        return False
        
    return verify_collection()

if __name__ == "__main__":
    success = main()
    if success:
        print("Collection verification complete")
        sys.exit(0)
    else:
        print("Failed to verify collection. See logs for details.")
        sys.exit(1)
