from pymilvus import connections, utility
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
    # Connect to Milvus (reusing from Step 1)
    logger.info("Connecting to Milvus server...")
    connections.connect(alias="default", host="localhost", port="19530")
    
    # Step 2: Validate Collection Existence
    logger.info(f"Step 2: Validating collection existence: '{COLLECTION_NAME}'")
    
    if utility.has_collection(COLLECTION_NAME):
        logger.info(f"✓ Collection '{COLLECTION_NAME}' exists")
        # Get additional collection information
        collection_stats = utility.get_collection_stats(COLLECTION_NAME)
        logger.info(f"✓ Collection info: {collection_stats}")
    else:
        logger.error(f"❌ Collection '{COLLECTION_NAME}' does not exist")
        logger.error("Aborting process: Target collection is missing")
        sys.exit(1)
        
except Exception as e:
    logger.error(f"Validation error: {str(e)}")
    sys.exit(1)

logger.info("Step 2 completed successfully.")
