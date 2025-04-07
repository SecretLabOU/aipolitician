#!/usr/bin/env python3
"""
Standalone script to list all politicians in the ChromaDB database

This script doesn't rely on imports from other files, making it more robust.
"""
import os
import sys
import argparse
import json
import logging
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chromadb-query")

# Default database path
DEFAULT_DB_PATH = os.path.expanduser("~/political_db")
DEFAULT_COLLECTION_NAME = "political_figures"

# Connect to ChromaDB
def connect_to_chroma(db_path: str = DEFAULT_DB_PATH) -> Optional[chromadb.Client]:
    """Connect to ChromaDB at the specified path"""
    try:
        # Create the client
        client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        logger.info(f"Connected to ChromaDB at {db_path}")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to ChromaDB: {e}")
        return None

# Get collection from ChromaDB
def get_collection(client: chromadb.Client, collection_name: str = DEFAULT_COLLECTION_NAME) -> Any:
    """Get a collection from ChromaDB"""
    try:
        # Check if collection exists
        collections = client.list_collections()
        
        # Convert to collection names if needed
        collection_names = [c.name if hasattr(c, 'name') else c for c in collections]
        
        if collection_name in collection_names:
            # Get existing collection
            collection = client.get_collection(name=collection_name)
            logger.info(f"Got collection: {collection_name}")
            return collection
        else:
            logger.error(f"Collection '{collection_name}' does not exist")
            return None
    except Exception as e:
        logger.error(f"Error getting collection: {e}")
        return None

# Get all politicians from the collection
def get_all_politicians(collection) -> List[Dict[str, Any]]:
    """Get all politicians from the collection"""
    try:
        # Get all politicians (limited to 10,000 for safety)
        result = collection.get(
            limit=10000,
            include=["metadatas"]
        )
        
        if result and 'metadatas' in result and result['metadatas']:
            logger.info(f"Retrieved {len(result['metadatas'])} politicians")
            return result['metadatas']
        else:
            logger.warning("No politicians found in collection")
            return []
    except Exception as e:
        logger.error(f"Failed to get all politicians: {e}")
        return []

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="List all politicians in the database")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, 
                        help=f"Path to ChromaDB database (default: {DEFAULT_DB_PATH})")
    parser.add_argument("--detailed", "-d", action="store_true", 
                        help="Show detailed information")
    
    args = parser.parse_args()
    
    try:
        # Connect to database
        print(f"Connecting to database at {args.db_path}...")
        client = connect_to_chroma(args.db_path)
        
        if not client:
            print(f"Error: Could not connect to ChromaDB at {args.db_path}")
            sys.exit(1)
            
        collection = get_collection(client)
        
        if not collection:
            print("Error: Could not get collection from ChromaDB")
            sys.exit(1)
        
        # List all politicians
        print("Retrieving all politicians...")
        all_politicians = get_all_politicians(collection)
        
        if not all_politicians:
            print("No politicians found in the database.")
            return
            
        print(f"\nFound {len(all_politicians)} politicians:\n")
        
        # Sort by name for easier viewing
        all_politicians.sort(key=lambda x: x.get('name', ''))
        
        for i, politician in enumerate(all_politicians, 1):
            print(f"{i}. {politician.get('name', 'Unknown')} " +
                  f"({politician.get('political_affiliation', 'Unknown')})")
            
            if args.detailed:
                print(f"   ID: {politician.get('id', 'Unknown')}")
                if 'biography' in politician:
                    bio = politician['biography']
                    print(f"   {bio[:100]}..." if bio else "   No biography")
                
                # Display policy data
                try:
                    if 'policies' in politician:
                        policies = json.loads(politician.get('policies', '{}'))
                        if policies:
                            print("   Policies:")
                            for area, positions in policies.items():
                                if positions:
                                    print(f"      {area.title()}: {', '.join(positions[:2])}" + 
                                        (f" (and {len(positions)-2} more...)" if len(positions) > 2 else ""))
                except:
                    pass
                    
                print("")
                
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
if __name__ == "__main__":
    main() 