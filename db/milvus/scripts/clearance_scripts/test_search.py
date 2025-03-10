from pymilvus import connections, Collection
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    logger.info("Connecting to Milvus...")
    connections.connect(host='localhost', port='19530')
    logger.info("Connected to Milvus successfully")
    
    logger.info("Loading political_figures collection...")
    collection = Collection('political_figures')
    collection.load()
    
    # Create a random embedding vector for testing
    logger.info("Creating test embedding vector...")
    test_embedding = np.random.random(768).astype(np.float32)
    
    # Search parameters
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 6}
    }
    
    logger.info("Performing vector search...")
    results = collection.search(
        data=[test_embedding], 
        anns_field="embedding", 
        param=search_params,
        limit=3,
        output_fields=["id", "name", "political_affiliation"]
    )
    
    print("\nSearch Results:")
    for i, hits in enumerate(results):
        print(f"Query {i+1}, results:")
        for j, hit in enumerate(hits):
            print(f"  Hit {j+1}: id={hit.id}, distance={hit.distance}, entity={hit.entity}")
except Exception as e:
    logger.error(f"Error: {e}")
