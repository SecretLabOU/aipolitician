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
    
    print("\nCollection Schema:")
    print(collection.schema)
    
    print("\nFields:")
    for field in collection.schema.fields:
        print(f"- {field.name}: {field.dtype} {'(primary key)' if field.is_primary else ''}")
except Exception as e:
    logger.error(f"Error: {e}")
