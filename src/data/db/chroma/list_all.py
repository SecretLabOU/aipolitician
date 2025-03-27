#!/usr/bin/env python3
"""
ChromaDB List All Politicians

A simple wrapper around query.py to list all politicians in the database.

Usage:
    python list_all.py
    python list_all.py --detailed
"""

import sys
import os
import argparse

# Determine the path to query.py
script_dir = os.path.dirname(os.path.abspath(__file__))
query_script = os.path.join(script_dir, "query.py")

def main():
    parser = argparse.ArgumentParser(description="List all politicians in the database")
    parser.add_argument("--db-path", default=os.path.expanduser("~/political_db"), 
                        help="Path to ChromaDB database")
    parser.add_argument("--detailed", "-d", action="store_true", 
                        help="Show detailed information")
    
    args = parser.parse_args()
    
    # Build command to run query.py with --list-all
    cmd = [
        sys.executable,
        query_script,
        "--list-all",
        "--db-path", args.db_path
    ]
    
    if args.detailed:
        cmd.append("--detailed")
        
    # Use os.execv to replace the current process with query.py
    os.execv(sys.executable, cmd)
    
if __name__ == "__main__":
    main() 