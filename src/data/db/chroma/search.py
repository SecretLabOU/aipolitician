#!/usr/bin/env python3
"""
ChromaDB Search Utility

This script provides a command-line interface to search the political figures 
database using ChromaDB's semantic search capabilities.

Usage:
    python search.py --query "climate change policy"
    python search.py --query "healthcare" --results 10
    python search.py --query "foreign policy" --db-path /path/to/db
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any

# Add the parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# Import ChromaDB modules
from data.db.chroma.schema import connect_to_chroma, get_collection, DEFAULT_DB_PATH
from data.db.chroma.operations import search_politicians, get_all_politicians

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_search_results(results: List[Dict[str, Any]], verbose: bool = False) -> None:
    """
    Print search results in a readable format.
    
    Args:
        results: List of search results
        verbose: Whether to print detailed information
    """
    if not results:
        print("No results found.")
        return
        
    print(f"\nFound {len(results)} results:\n")
    
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.get('name', 'Unknown')} " +
              f"({result.get('political_affiliation', 'Unknown affiliation')})")
        
        if 'distance' in result:
            print(f"   Relevance: {100 * (1 - result['distance']):.2f}%")
            
        print(f"   {result.get('biography', '')}")
        
        if verbose:
            # Print more detailed information if verbose mode is enabled
            print("\n   Key information:")
            
            if result.get('date_of_birth'):
                print(f"   • Born: {result.get('date_of_birth')}")
                
            if result.get('nationality'):
                print(f"   • Nationality: {result.get('nationality')}")
                
            # Parse and print policies if available
            try:
                policies = json.loads(result.get('policies', '{}'))
                if policies:
                    print("\n   Policy positions:")
                    for area, positions in policies.items():
                        if positions:
                            print(f"   • {area.title()}: {positions[0]}" + 
                                  (f" (and {len(positions)-1} more...)" if len(positions) > 1 else ""))
            except:
                pass
                
        print("\n" + "-" * 50)

def main():
    """Main entry point for the search script."""
    parser = argparse.ArgumentParser(description="Search the political figures database")
    
    parser.add_argument("--query", help="Search query", required=True)
    parser.add_argument("--results", type=int, default=5, help="Number of results to return")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, 
                       help=f"Path to ChromaDB database (default: {DEFAULT_DB_PATH})")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed results")
    parser.add_argument("--list-all", action="store_true", help="List all politicians in the database")
    
    args = parser.parse_args()
    
    try:
        # Connect to ChromaDB
        logger.info(f"Connecting to ChromaDB at {args.db_path}...")
        client = connect_to_chroma(db_path=args.db_path)
        if not client:
            logger.error("Failed to connect to ChromaDB")
            print("Error: Could not connect to the database.")
            sys.exit(1)
            
        # Get collection
        collection = get_collection(client)
        if not collection:
            logger.error("Failed to get collection")
            print("Error: Could not access the politicians collection.")
            sys.exit(1)
            
        if args.list_all:
            # List all politicians in the database
            logger.info("Listing all politicians...")
            all_politicians = get_all_politicians(collection)
            
            if all_politicians:
                print(f"\nFound {len(all_politicians)} politicians in the database:\n")
                for i, politician in enumerate(sorted(all_politicians, key=lambda x: x.get('name', '')), 1):
                    print(f"{i}. {politician.get('name', 'Unknown')} " +
                          f"({politician.get('political_affiliation', 'Unknown')})")
            else:
                print("No politicians found in the database.")
                
        else:
            # Search for politicians
            logger.info(f"Searching for: {args.query}")
            results = search_politicians(collection, args.query, n_results=args.results)
            
            # Print results
            print_search_results(results, verbose=args.verbose)
            
    except Exception as e:
        logger.error(f"Error during search: {e}")
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 