from pymilvus import connections, Collection
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Milvus-Clearance")

# Collection name 
COLLECTION_NAME = "political_figures"

try:
    # Connect to Milvus
    logger.info("Connecting to Milvus server...")
    connections.connect(alias="default", host="localhost", port="19530")
    
    # Get collection
    collection = Collection(COLLECTION_NAME)
    collection.load()
    
    # Double-check entity count
    entity_count = collection.num_entities
    logger.info(f"Entity count verification: {entity_count} entities")
    
    # Try to query some data if possible
    if entity_count > 0:
        logger.info("Attempting to query sample data...")
        results = collection.query(expr="id != ''", output_fields=["id", "name"], limit=5)
        logger.info(f"Query returned {len(results)} results:")
        for i, result in enumerate(results):
            logger.info(f"  Result {i+1}: {result}")
    else:
        logger.info("Collection is already empty. No data to delete.")
    
except Exception as e:
    logger.error(f"Error during entity verification: {str(e)}")
    sys.exit(1)

logger.info("Entity verification completed.")
