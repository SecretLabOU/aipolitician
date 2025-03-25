#!/usr/bin/env python3
"""
Database management script for the Milvus vector database.
Allows creating, listing, and dropping collections.
"""
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add project root to path
root_dir = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(root_dir))

try:
    from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType
    from src.data.db.milvus.connection import get_connection_params
except ImportError:
    print("Error: Required packages not found. Please install with:")
    print("pip install -r requirements/requirements-langgraph.txt")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_VECTOR_DIM = 768  # Default dimension for embeddings
DEFAULT_NLIST = 1024      # Default IVF parameter
DEFAULT_NPROBE = 16       # Default number of probe vectors


def create_collection(name: str, dim: int = DEFAULT_VECTOR_DIM) -> Collection:
    """
    Create a new collection in Milvus.
    
    Args:
        name: Name of the collection
        dim: Dimension of the embeddings
        
    Returns:
        Created collection
    """
    # Define fields
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="politician", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="date", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="topics", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_length=100, max_capacity=20),
        FieldSchema(name="policy_areas", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_length=100, max_capacity=20),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    
    # Create schema
    schema = CollectionSchema(fields=fields, description=f"Political knowledge for AI Politician - {name}")
    
    # Create collection
    collection = Collection(name=name, schema=schema)
    
    # Create index on embeddings
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": DEFAULT_NLIST}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    
    return collection


def list_collections() -> List[str]:
    """
    List all collections in Milvus.
    
    Returns:
        List of collection names
    """
    return utility.list_collections()


def drop_collection(name: str) -> bool:
    """
    Drop a collection from Milvus.
    
    Args:
        name: Name of the collection to drop
        
    Returns:
        True if successful, False otherwise
    """
    if utility.has_collection(name):
        utility.drop_collection(name)
        return True
    return False


def get_collection_stats(name: str) -> Dict[str, Any]:
    """
    Get statistics for a collection.
    
    Args:
        name: Name of the collection
        
    Returns:
        Dictionary with collection statistics
    """
    if not utility.has_collection(name):
        return {"error": f"Collection '{name}' does not exist"}
    
    collection = Collection(name)
    collection.load()
    
    try:
        # Get entity count
        stats = {"name": name, "count": collection.num_entities}
        
        # Get collection info
        stats["created"] = "Unknown"  # Milvus doesn't provide creation time directly
        
        # Get index info
        index_infos = collection.index().params
        stats["indexes"] = index_infos
        
        return stats
    finally:
        collection.release()


def search_collection(name: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Perform a simple text search in a collection.
    
    Args:
        name: Name of the collection
        query: Text query
        limit: Maximum number of results
        
    Returns:
        List of search results
    """
    if not utility.has_collection(name):
        return [{"error": f"Collection '{name}' does not exist"}]
    
    # In a real implementation, this would use an embedding model
    # For simplicity, we're just doing a simple example here
    return [{"warning": "Search not implemented in this script"}]


def main():
    """Main function handling command-line arguments."""
    parser = argparse.ArgumentParser(description="Milvus Database Management for AI Politician")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new collection")
    create_parser.add_argument("--collection", required=True, help="Collection name")
    create_parser.add_argument("--dim", type=int, default=DEFAULT_VECTOR_DIM,
                            help=f"Vector dimension (default: {DEFAULT_VECTOR_DIM})")
    
    # List command
    subparsers.add_parser("list", help="List all collections")
    
    # Count command
    count_parser = subparsers.add_parser("count", help="Count entities in a collection")
    count_parser.add_argument("--collection", required=True, help="Collection name")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Get collection statistics")
    stats_parser.add_argument("--collection", required=True, help="Collection name")
    
    # Drop command
    drop_parser = subparsers.add_parser("drop", help="Drop a collection")
    drop_parser.add_argument("--collection", required=True, help="Collection name")
    drop_parser.add_argument("--force", action="store_true", help="Force drop without confirmation")
    
    # Search command (simplified example)
    search_parser = subparsers.add_parser("search", help="Search for content in a collection")
    search_parser.add_argument("--collection", required=True, help="Collection name")
    search_parser.add_argument("--query", required=True, help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Maximum number of results")
    
    # Verbose flag
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Get connection parameters
    try:
        conn_params = get_connection_params()
        connections.connect(**conn_params)
        logger.info(f"Connected to Milvus at {conn_params.get('host')}:{conn_params.get('port')}")
    except Exception as e:
        logger.error(f"Failed to connect to Milvus: {str(e)}")
        sys.exit(1)
    
    try:
        # Execute the specified command
        if args.command == "create":
            if utility.has_collection(args.collection):
                logger.warning(f"Collection '{args.collection}' already exists")
            else:
                collection = create_collection(args.collection, args.dim)
                logger.info(f"Created collection '{args.collection}' with dimension {args.dim}")
        
        elif args.command == "list":
            collections = list_collections()
            if collections:
                logger.info("Collections:")
                for i, coll_name in enumerate(collections, 1):
                    # Get entity count if possible
                    try:
                        count = Collection(coll_name).num_entities
                        logger.info(f"  {i}. {coll_name} ({count} entities)")
                    except:
                        logger.info(f"  {i}. {coll_name}")
            else:
                logger.info("No collections found")
        
        elif args.command == "count":
            if utility.has_collection(args.collection):
                collection = Collection(args.collection)
                count = collection.num_entities
                logger.info(f"Collection '{args.collection}' contains {count} entities")
            else:
                logger.error(f"Collection '{args.collection}' does not exist")
        
        elif args.command == "stats":
            if utility.has_collection(args.collection):
                stats = get_collection_stats(args.collection)
                logger.info(f"Statistics for collection '{args.collection}':")
                logger.info(f"  • Entities: {stats['count']}")
                logger.info(f"  • Index type: {stats['indexes'].get('index_type', 'unknown')}")
            else:
                logger.error(f"Collection '{args.collection}' does not exist")
        
        elif args.command == "drop":
            if utility.has_collection(args.collection):
                if not args.force:
                    confirm = input(f"Are you sure you want to drop collection '{args.collection}'? (y/N): ")
                    if confirm.lower() != 'y':
                        logger.info("Operation cancelled")
                        return
                
                if drop_collection(args.collection):
                    logger.info(f"Dropped collection '{args.collection}'")
                else:
                    logger.error(f"Failed to drop collection '{args.collection}'")
            else:
                logger.error(f"Collection '{args.collection}' does not exist")
        
        elif args.command == "search":
            if utility.has_collection(args.collection):
                logger.warning("Search is not fully implemented in this script")
                logger.info(f"Would search for '{args.query}' in collection '{args.collection}'")
            else:
                logger.error(f"Collection '{args.collection}' does not exist")
        
        else:
            parser.print_help()
    
    finally:
        # Disconnect from Milvus
        connections.disconnect(conn_params.get("alias", "default"))
        logger.debug("Disconnected from Milvus")


if __name__ == "__main__":
    main() 