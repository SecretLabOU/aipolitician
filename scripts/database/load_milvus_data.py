#!/usr/bin/env python3
"""
Script to load processed data into the Milvus database for the AI Politician system.
"""
import sys
import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path
root_dir = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(root_dir))

try:
    from pymilvus import connections, utility, Collection
    from src.data.db.milvus.connection import get_connection_params, get_collection_name
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

def load_processed_data(data_path: str) -> List[Dict[str, Any]]:
    """
    Load processed data from JSON files.
    
    Args:
        data_path: Path to directory containing processed data
        
    Returns:
        List of data entries
    """
    data = []
    path = Path(data_path)
    
    if not path.exists():
        logger.error(f"Path does not exist: {path}")
        return []
    
    if path.is_file() and path.suffix.lower() == '.json':
        # Single file
        try:
            with open(path, 'r') as f:
                entries = json.load(f)
                if isinstance(entries, list):
                    data.extend(entries)
                else:
                    data.append(entries)
            logger.info(f"Loaded {len(data)} entries from {path}")
        except Exception as e:
            logger.error(f"Error loading {path}: {str(e)}")
    else:
        # Directory of files
        json_files = list(path.glob('**/*.json'))
        logger.info(f"Found {len(json_files)} JSON files in {path}")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    entries = json.load(f)
                    if isinstance(entries, list):
                        data.extend(entries)
                    else:
                        data.append(entries)
                logger.info(f"Loaded data from {json_file}")
            except Exception as e:
                logger.error(f"Error loading {json_file}: {str(e)}")
    
    return data

def main():
    """Main function to load data into Milvus."""
    parser = argparse.ArgumentParser(description="Load processed data into Milvus for AI Politician")
    parser.add_argument("--path", required=True, help="Path to directory with processed data")
    parser.add_argument("--politician", choices=["biden", "trump"], help="Politician name for collection selection")
    parser.add_argument("--collection", help="Override collection name")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Get connection parameters
    conn_params = get_connection_params()
    
    # Connect to Milvus
    try:
        connections.connect(**conn_params)
        logger.info(f"Connected to Milvus server at {conn_params.get('host')}:{conn_params.get('port')}")
    except Exception as e:
        logger.error(f"Failed to connect to Milvus: {str(e)}")
        sys.exit(1)
    
    # Get collection name
    collection_name = args.collection
    if not collection_name:
        collection_name = get_collection_name(args.politician)
    
    # Check if collection exists
    if not utility.has_collection(collection_name):
        logger.error(f"Collection '{collection_name}' does not exist. Please create it first.")
        sys.exit(1)
    
    # Load data
    data = load_processed_data(args.path)
    if not data:
        logger.error("No data loaded. Exiting.")
        sys.exit(1)
    
    logger.info(f"Loaded {len(data)} entries in total")
    
    # Insert into collection
    try:
        collection = Collection(collection_name)
        collection.load()
        
        # Insert data
        result = collection.insert(data)
        logger.info(f"Inserted {result.insert_count} entries into collection '{collection_name}'")
        
        # Flush to ensure data is committed
        collection.flush()
        logger.info("Data flushed to disk")
        
    except Exception as e:
        logger.error(f"Error inserting data: {str(e)}")
        sys.exit(1)
    finally:
        # Disconnect from Milvus
        connections.disconnect(conn_params.get("alias", "default"))
        logger.info("Disconnected from Milvus")

if __name__ == "__main__":
    main()
