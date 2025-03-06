#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

#!/usr/bin/env python3
import sys
import logging
from sentence_transformers import SentenceTransformer
from pymilvus import Collection, connections

# Configure logging
logging.basicConfig(level=logging.INFO)
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

def search_politicians(query, limit=3):
    """Search for politicians matching the query."""
    # Connect to Milvus
    if not connect_to_milvus():
        logger.error("Failed to connect to Milvus")
        return
    
    # Get the collection
    collection = Collection(name="political_figures")
    collection.load()
    
    # Load the embedding model
    model = SentenceTransformer('all-mpnet-base-v2')  # Using model that produces 768-dim vectors
    
    # Generate embedding for query
    query_embedding = model.encode(query).tolist()
    
    # Define search parameters
    search_params = {
        "metric_type": "L2",
        "params": {"ef": 100}
    }
    
    # Execute search
    results = collection.search(
        data=[query_embedding], 
        anns_field="embedding", 
        param=search_params,
        limit=limit,
        output_fields=["id", "name", "political_affiliation", "biography"]
    )
    
    # Process results
    search_results = []
    for hits in results:
        for hit in hits:
            result = {
                "id": hit.entity.get("id"),
                "name": hit.entity.get("name"),
                "political_affiliation": hit.entity.get("political_affiliation"),
                "biography": hit.entity.get("biography"),
                "similarity": hit.score
            }
            search_results.append(result)
            
    return search_results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_search.py <search_query>")
        sys.exit(1)
        
    query = sys.argv[1]
    print(f"Searching for: {query}")
    
    results = search_politicians(query)
    
    if not results:
        print("No results found.")
    else:
        print(f"\nFound {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"Name: {result['name']}")
            print(f"Political Affiliation: {result['political_affiliation']}")
            print(f"Similarity Score: {result['similarity']:.4f}")
            print(f"Biography: {result['biography'][:200]}...")
