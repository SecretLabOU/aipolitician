from pymilvus import connections, utility
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Milvus-Clearance")

try:
    # Step 1: Establish Milvus Connection
    logger.info("Step 1: Establishing connection to Milvus server...")
    connections.connect(
        alias="default", 
        host="localhost",
        port="19530"
    )
    
    # Validate connection through server heartbeat
    if utility.get_server_version():
        logger.info(f"✓ Successfully connected to Milvus server")
        logger.info(f"✓ Server version: {utility.get_server_version()}")
    else:
        logger.error("Failed to get server version. Connection may be unstable.")
        sys.exit(1)
        
except Exception as e:
    logger.error(f"Connection error: {str(e)}")
    sys.exit(1)

logger.info("Step 1 completed successfully.")
