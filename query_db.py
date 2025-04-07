#!/usr/bin/env python3
"""
ChromaDB Query Tool

This script provides a simple command-line interface to query the political figures database.
Run this from the project root to avoid import issues.

Usage:
    python query_db.py --query "Donald Trump" --detailed
    python query_db.py --id "<politician-id>" --detailed
"""

import os
import sys
import argparse
import json

# Add the project root directory to the Python path
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

# Import the required modules with absolute imports
try:
    from src.data.db.chroma.operations import search_politicians, get_politician_by_id
    from src.data.db.chroma.schema import connect_to_chroma, get_collection, DEFAULT_COLLECTION_NAME
    print("Successfully imported modules")
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def format_politician(politician, detailed=False):
    """Format politician data for display"""
    output = [f"Name: {politician.get('name', 'Unknown')}"]
    output.append(f"Political Affiliation: {politician.get('political_affiliation', 'Unknown')}")
    output.append(f"Biography: {politician.get('biography', 'N/A')[:200]}...")
    
    if detailed:
        output.append("\nPolicy Positions:")
        try:
            policies = json.loads(politician.get('policies', '{}'))
            for area, positions in policies.items():
                if positions:
                    output.append(f"  {area.title()}: {', '.join(positions[:2])}" + 
                                 (f" (and {len(positions)-2} more...)" if len(positions) > 2 else ""))
        except Exception as e:
            output.append(f"  Error parsing policy data: {e}")
            
        if 'id' in politician:
            output.append(f"\nID: {politician['id']}")
            
    return "\n".join(output)

def main():
    parser = argparse.ArgumentParser(description="Query the politicians database")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", help="Search query")
    group.add_argument("--id", help="Get politician by ID")
    
    parser.add_argument("--db-path", default=os.path.expanduser("~/political_db"), 
                        help="Path to ChromaDB database")
    parser.add_argument("--results", type=int, default=5, help="Number of results to return")
    parser.add_argument("--detailed", "-d", action="store_true", help="Show detailed information")
    
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
        
        if args.query:
            # Search by query
            print(f"Searching for: {args.query}")
            results = search_politicians(collection, args.query, n_results=args.results)
            
            if not results:
                print("No results found.")
                return
                
            print(f"\nFound {len(results)} results:\n")
            for i, result in enumerate(results, 1):
                print(f"Result {i}:")
                print(format_politician(result, args.detailed))
                print("-" * 50)
                
        elif args.id:
            # Get by ID
            print(f"Looking up politician with ID: {args.id}")
            politician = get_politician_by_id(collection, args.id)
            
            if not politician:
                print(f"No politician found with ID: {args.id}")
                return
                
            print(format_politician(politician, detailed=True))
                
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    main()
