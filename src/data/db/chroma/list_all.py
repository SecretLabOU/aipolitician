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

# Add the project root to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../../.."))
sys.path.insert(0, project_root)

# Import local modules
try:
    # Try direct import first
    sys.path.insert(0, script_dir)
    from schema import connect_to_chroma, get_collection
    from operations import get_all_politicians
except ImportError:
    # Try absolute imports if direct imports fail
    try:
        from src.data.db.chroma.schema import connect_to_chroma, get_collection
        from src.data.db.chroma.operations import get_all_politicians
    except ImportError:
        print("Error: Could not import required modules. Make sure you're running this from the project root.")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Script directory: {script_dir}")
        print(f"Project root: {project_root}")
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
        import traceback
        traceback.print_exc()
    
if __name__ == "__main__":
    main() 