#!/usr/bin/env python3
import sys
import os
import logging
import traceback
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

#!/usr/bin/env python3
"""
Initialize the Milvus database for AI Politician project.
This script sets up the Milvus collection and indexes.
"""

import os
import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent
sys.path.append(str(parent_dir))

from scripts.schema import connect_to_milvus, create_political_figures_collection, create_hnsw_index

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(current_dir.parent, "logs", "initialize_db.log"))
    ]
)

logger = logging.getLogger(__name__)

def wait_for_milvus_ready(max_attempts=30, wait_time=5):
    """Wait for Milvus server to be ready, with retry"""
    logger.info("Waiting for Milvus server to be ready...")
    
    for attempt in range(max_attempts):
        try:
            if connect_to_milvus():
                logger.info("Milvus server is ready")
                return True
            else:
                logger.warning(f"Milvus not ready yet. Attempt {attempt+1}/{max_attempts}")
                time.sleep(wait_time)
        except Exception as e:
            logger.warning(f"Error connecting to Milvus: {str(e)}. Attempt {attempt+1}/{max_attempts}")
            time.sleep(wait_time)
    
    logger.error(f"Milvus server not ready after {max_attempts} attempts")
    return False

def initialize_database(recreate=False):
    """Initialize the database with collections and indexes"""
    logger.info("Initializing Milvus database...")
    
    # Wait for Milvus to be ready
    if not wait_for_milvus_ready():
        logger.error("Failed to connect to Milvus. Please check if Milvus is running.")
        return False
    
    try:
        # Create collection
        collection = create_political_figures_collection(drop_existing=recreate)
        logger.info(f"Collection ready: {collection.name}")
        
        # Create index if needed
        if recreate or not collection.has_index():
            create_hnsw_index(collection.name)
            logger.info("Index created successfully")
        else:
            logger.info("Index already exists")
        
        logger.info("Database initialization completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize Milvus database for AI Politician")
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate existing collections")
    args = parser.parse_args()
    
    success = initialize_database(recreate=args.recreate)
    
    if success:
        print("Database initialization completed successfully")
        sys.exit(0)
    else:
        print("Database initialization failed. See logs for details.")
        sys.exit(1)
