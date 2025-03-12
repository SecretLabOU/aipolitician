#!/usr/bin/env python3
"""
Clear all data from the 'political_figures' collection while preserving its structure.
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

def clear_collection_data(collection_name="political_figures"):
    """
    Clear all data from a collection while preserving its structure.
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
        
        # Get the number of entities before clearing
        entity_count_before = collection.num_entities
        logger.info(f"Collection '{collection_name}' has {entity_count_before} entities before clearing")
        
        if entity_count_before == 0:
            logger.info(f"Collection '{collection_name}' is already empty")
            return True
            
        # Try alternative approaches to delete all entities
        
        # First, try the simplest approach
        try:
            logger.info(f"Attempting to delete all entities using simple expression...")
            collection.delete("id != ''")
            collection.flush()
            
            # Check if that worked
            entity_count_after_simple = collection.num_entities
            if entity_count_after_simple == 0:
                logger.info(f"Successfully cleared all data using simple expression")
                return True
            else:
                logger.info(f"Simple expression deleted some but not all entities. Trying alternative method...")
        except Exception as e:
            logger.warning(f"Simple deletion approach failed: {str(e)}. Trying alternative method...")
        
        # Get the primary key field name from the schema
        schema = collection.schema
        primary_field = next((field for field in schema.fields if field.is_primary), None)
        
        if not primary_field:
            logger.error("Could not find primary key field in collection schema")
            return False
            
        primary_field_name = primary_field.name
        logger.info(f"Primary key field is '{primary_field_name}'")
        
        # Query all primary keys first
        logger.info(f"Querying all primary keys from collection '{collection_name}'...")
        results = collection.query(expr="", output_fields=[primary_field_name], limit=entity_count_before+100)
        
        if not results:
            logger.warning("No entities found in the collection when querying")
            return True
            
        logger.info(f"Found {len(results)} entities to delete")
        
        # Delete entities in batches if there are many
        BATCH_SIZE = 1000
        for i in range(0, len(results), BATCH_SIZE):
            batch = results[i:i+BATCH_SIZE]
            ids = [doc[primary_field_name] for doc in batch]
            
            # Convert the list of IDs to a comma-separated string of quoted IDs
            id_list_str = ", ".join([f"'{id}'" for id in ids])
            expr = f"{primary_field_name} in [{id_list_str}]"
            
            logger.info(f"Deleting batch {i//BATCH_SIZE + 1}/{(len(results)-1)//BATCH_SIZE + 1} with {len(batch)} entities...")
            
            try:
                collection.delete(expr)
                logger.info(f"Deleted batch {i//BATCH_SIZE + 1}")
            except Exception as e:
                logger.error(f"Error deleting batch: {str(e)}")
        
        # Flush to ensure changes are persisted
        collection.flush()
        
        # Get the number of entities after clearing
        entity_count_after = collection.num_entities
        logger.info(f"Collection '{collection_name}' has {entity_count_after} entities after clearing")
        
        if entity_count_after == 0:
            logger.info(f"Successfully cleared all data from collection '{collection_name}'")
            return True
        else:
            logger.warning(f"Some entities remain in collection '{collection_name}' after clearing")
            
            # Last resort: recreate the collection with same schema
            logger.info("Attempting to recreate collection with same schema...")
            
            # Save the schema details
            old_schema = collection.schema
            
            # Drop and recreate
            utility.drop_collection(collection_name)
            
            # Recreate with same schema
            from scripts.schema import create_political_figures_collection, create_hnsw_index
            create_political_figures_collection(drop_existing=False)
            create_hnsw_index(collection_name)
            
            logger.info(f"Collection '{collection_name}' recreated with the same schema")
            return True
            
    except Exception as e:
        logger.error(f"Error clearing collection data: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function."""
    if not connect_to_milvus():
        logger.error("Failed to connect to Milvus server")
        return False
        
    return clear_collection_data()

if __name__ == "__main__":
    success = main()
    if success:
        print("Collection data cleared successfully")
        sys.exit(0)
    else:
        print("Failed to clear collection data. See logs for details.")
        sys.exit(1)
