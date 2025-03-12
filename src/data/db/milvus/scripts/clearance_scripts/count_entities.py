from pymilvus import connections, Collection
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    logger.info("Connecting to Milvus...")
    connections.connect(host='localhost', port='19530')
    logger.info("Connected to Milvus successfully")
    
    logger.info("Loading political_figures collection...")
    collection = Collection('political_figures')
    
    # Get the number of entities
    entity_count = collection.num_entities
    print(f"\nTotal entities in collection: {entity_count}")
    
    # If collection has entities, retrieve a sample
    if entity_count > 0:
        logger.info("Retrieving sample data...")
        collection.load()
        results = collection.query(expr="id != ''", output_fields=["id", "name", "nationality", "political_affiliation"], limit=5)
        print("\nSample Data (first 5 entities):")
        for i, result in enumerate(results):
            print(f"Entity {i+1}:")
            for key, value in result.items():
                print(f"  {key}: {value}")
except Exception as e:
    logger.error(f"Error: {e}")
