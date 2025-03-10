from pymilvus import connections, Collection, utility
import logging
import sys
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Milvus-Clearance")

# Collection name 
COLLECTION_NAME = "political_figures"
MAX_RETRIES = 3
RETRY_DELAY = 2

try:
    # Connect to Milvus
    logger.info("Connecting to Milvus server...")
    connections.connect(alias="default", host="localhost", port="19530")
    
    # Step 2: Validate Collection Existence (again for this script)
    logger.info(f"Validating collection existence: '{COLLECTION_NAME}'")
    if not utility.has_collection(COLLECTION_NAME):
        logger.error(f"❌ Collection '{COLLECTION_NAME}' does not exist")
        sys.exit(1)
    
    # Step 3: Preload Collection Safely
    logger.info("Step 3: Preloading collection safely...")
    collection = Collection(COLLECTION_NAME)
    
    # Implement retry logic for loading
    retry_count = 0
    load_success = False
    
    while retry_count < MAX_RETRIES and not load_success:
        try:
            logger.info(f"Loading collection (attempt {retry_count + 1}/{MAX_RETRIES})...")
            collection.load()
            load_success = True
            logger.info("✓ Collection loaded successfully")
        except Exception as e:
            retry_count += 1
            logger.warning(f"Load attempt {retry_count} failed: {str(e)}")
            if retry_count < MAX_RETRIES:
                logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                logger.error("Maximum retry attempts reached. Aborting.")
                raise
    
    # Step 4: Dataset Analysis
    logger.info("Step 4: Analyzing dataset...")
    entity_count = collection.num_entities
    logger.info(f"✓ Collection currently contains {entity_count} entities")
    
    # Get more collection information if possible
    logger.info(f"Collection name: {collection.name}")
    logger.info(f"Collection description: {collection.description}")
    logger.info(f"Collection schema: {collection.schema}")
    
except Exception as e:
    logger.error(f"Error during collection preload or analysis: {str(e)}")
    sys.exit(1)

logger.info("Steps 3 and 4 completed successfully.")
