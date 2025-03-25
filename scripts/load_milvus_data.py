#!/usr/bin/env python3
"""
Milvus Database Data Loader

This script loads political figure data into the Milvus database using the pipeline.
It ensures the database is properly initialized and provides a simple interface
for loading data for specific politicians.
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(root_dir))

try:
    from src.data.pipeline.pipeline import run_pipeline
    from src.data.db.milvus.scripts.schema import initialize_database
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure you're running this script from the project root directory.")
    sys.exit(1)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Load politician data into Milvus database")
    
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--politicians', type=str, help='Comma-separated list of politicians to load data for')
    group.add_argument('--file', type=str, help='File containing one politician name per line')
    
    parser.add_argument('--force-reload', action='store_true', 
                      help='Force reload data even if it already exists')
    
    return parser.parse_args()

def read_politicians_from_file(file_path):
    """Read politician names from a file, one per line"""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

async def load_data(args):
    """Load data into Milvus database"""
    # Initialize database
    print("Initializing Milvus database...")
    collection = initialize_database()
    
    # Get list of politicians
    politicians = []
    if args.politicians:
        politicians = [name.strip() for name in args.politicians.split(',')]
    elif args.file:
        politicians = read_politicians_from_file(args.file)
    else:
        # Default to these politicians if none specified
        politicians = ["Joe Biden", "Donald Trump", "Kamala Harris"]
    
    print(f"Loading data for {len(politicians)} politicians: {', '.join(politicians)}")
    
    # Run the pipeline
    stats = await run_pipeline(politicians)
    
    # Print results
    print("\nData Loading Results:")
    print(f"Total processed: {stats['total']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    
    if stats['failed'] > 0:
        print("\nFailed politicians:")
        for name in stats['failures']:
            print(f"- {name}")
    
    print("\nData loading complete!")
    return stats

def main():
    """Main entry point for data loading script"""
    try:
        args = parse_arguments()
        asyncio.run(load_data(args))
    except Exception as e:
        print(f"Error during data loading: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
