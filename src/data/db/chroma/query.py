#!/usr/bin/env python3
"""
ChromaDB Query Tool

This script provides a simple command-line interface to query the political figures database.

Usage:
    python query.py --query "Donald Trump"
    python query.py --list-all
    python query.py --id <politician-id>
"""

import sys
import os
import argparse
import json
import importlib.util
import traceback

# Print useful debugging information
print(f"Current working directory: {os.getcwd()}")
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Script directory: {script_dir}")

# Try multiple approaches to import the required modules
connect_to_chroma = None
get_collection = None
search_politicians = None
get_politician_by_id = None
get_all_politicians = None

# Function to import module by path
def import_module_from_file(module_name, file_path):
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if not spec:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"Error importing {module_name} from {file_path}: {e}")
        return None

# Approach 1: Try direct import with full paths
schema_path = os.path.join(script_dir, "schema.py")
operations_path = os.path.join(script_dir, "operations.py")

schema_module = import_module_from_file("schema", schema_path)
operations_module = import_module_from_file("operations", operations_path)

if schema_module and operations_module:
    connect_to_chroma = schema_module.connect_to_chroma
    get_collection = schema_module.get_collection
    search_politicians = operations_module.search_politicians
    get_politician_by_id = operations_module.get_politician_by_id
    get_all_politicians = operations_module.get_all_politicians
    print("Successfully imported modules directly from files")

# Approach 2: Add script_dir to path and try regular imports
if not connect_to_chroma:
    try:
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        from schema import connect_to_chroma, get_collection
        from operations import search_politicians, get_politician_by_id, get_all_politicians
        print("Successfully imported modules using script_dir path")
    except ImportError as e:
        print(f"Import error using script_dir: {e}")

# Approach 3: Try absolute imports
if not connect_to_chroma:
    try:
        project_root = os.path.abspath(os.path.join(script_dir, "../../../.."))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from src.data.db.chroma.schema import connect_to_chroma, get_collection
        from src.data.db.chroma.operations import search_politicians, get_politician_by_id, get_all_politicians
        print("Successfully imported modules using absolute imports")
    except ImportError as e:
        print(f"Import error using absolute imports: {e}")

# Check if imports were successful
if not all([connect_to_chroma, get_collection, search_politicians, get_politician_by_id, get_all_politicians]):
    print("Failed to import required modules after trying multiple approaches")
    print(f"Python path: {sys.path}")
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
        except:
            output.append("  No policy data available")
            
        if 'id' in politician:
            output.append(f"\nID: {politician['id']}")
            
    return "\n".join(output)

def main():
    parser = argparse.ArgumentParser(description="Query the political figures database")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", help="Search query")
    group.add_argument("--list-all", action="store_true", help="List all politicians")
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
                
        elif args.list_all:
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
                    print(f"   {politician.get('biography', '')[:100]}...")
                    print("")
                    
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
        traceback.print_exc()
        
if __name__ == "__main__":
    main() 