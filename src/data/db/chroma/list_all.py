#!/usr/bin/env python3
"""
ChromaDB List All Politicians

A simple script to list all politicians in the database.

Usage:
    python list_all.py
    python list_all.py --detailed
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
    get_all_politicians = operations_module.get_all_politicians
    print("Successfully imported modules directly from files")

# Approach 2: Add script_dir to path and try regular imports
if not connect_to_chroma:
    try:
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        from schema import connect_to_chroma, get_collection
        from operations import get_all_politicians
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
        from src.data.db.chroma.operations import get_all_politicians
        print("Successfully imported modules using absolute imports")
    except ImportError as e:
        print(f"Import error using absolute imports: {e}")

# Check if imports were successful
if not all([connect_to_chroma, get_collection, get_all_politicians]):
    print("Failed to import required modules after trying multiple approaches")
    print(f"Python path: {sys.path}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="List all politicians in the database")
    parser.add_argument("--db-path", default=os.path.expanduser("~/political_db"), 
                        help="Path to ChromaDB database")
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
                print("")
                
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
    
if __name__ == "__main__":
    main() 