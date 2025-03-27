#!/usr/bin/env python3
"""
ChromaDB Database Initialization Script

This script initializes the ChromaDB database for the AI Politician project.
It creates the database directory with proper permissions and sets up the collection.

Usage:
    python initialize_db.py [--db-path /path/to/db]
"""

import os
import sys
import argparse
from pathlib import Path

# Add the parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# Import ChromaDB modules
from data.db.chroma.schema import initialize_database, DEFAULT_DB_PATH, setup_permissions

def main():
    """Initialize the ChromaDB database."""
    parser = argparse.ArgumentParser(description="Initialize ChromaDB database")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, 
                        help=f"Path to ChromaDB database (default: {DEFAULT_DB_PATH})")
    args = parser.parse_args()
    
    # Convert to absolute path
    db_path = os.path.abspath(os.path.expanduser(args.db_path))
    
    print(f"Initializing ChromaDB database at: {db_path}")
    
    # Ensure the directory exists with proper permissions
    setup_permissions(db_path)
    
    # Initialize the database
    try:
        db = initialize_database(db_path=db_path)
        print(f"Database initialized successfully!")
        print(f"Collection name: {db['collection'].name}")
        print(f"Database path: {db_path}")
        
        # Print permissions info
        print("\nPermission information:")
        path = Path(db_path)
        if path.exists():
            mode = path.stat().st_mode
            permissions = f"{mode & 0o777:o}"
            print(f"Directory permissions: {permissions}")
            print("This means:")
            print(f"  - Owner: {'r' if mode & 0o400 else '-'}{'w' if mode & 0o200 else '-'}{'x' if mode & 0o100 else '-'}")
            print(f"  - Group: {'r' if mode & 0o40 else '-'}{'w' if mode & 0o20 else '-'}{'x' if mode & 0o10 else '-'}")
            print(f"  - Other: {'r' if mode & 0o4 else '-'}{'w' if mode & 0o2 else '-'}{'x' if mode & 0o1 else '-'}")
        
        print("\nThe database is now ready to use.")
        print("You can populate it using the pipeline script:")
        print("python src/data/pipeline/pipeline.py --file politicians.txt")
        
    except Exception as e:
        print(f"Error initializing database: {e}")
        sys.exit(1)
    
if __name__ == "__main__":
    main() 