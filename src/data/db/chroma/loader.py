#!/usr/bin/env python3
"""
Load Politicians Data into ChromaDB

This script loads politician data from JSON files produced by the scraper
into the ChromaDB database.

Usage:
    python -m src.data.db.chroma.loader --source-dir /path/to/json/files --db-path /path/to/db

The script will process all JSON files in the source directory and load each entry
into the ChromaDB database. It handles multiple entries per JSON file (unlike the pipeline
which might only load the first entry).
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import schema and operations from the same package
from .schema import connect_to_chroma, get_collection, DEFAULT_DB_PATH
from .operations import upsert_politician

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chroma_loader.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("chroma_loader")

def normalize_politician_data(data: Dict[str, Any], filename: str) -> Dict[str, Any]:
    """
    Normalize the politician data from different formats into a consistent format.
    
    Args:
        data: Raw politician data
        filename: Source filename (used for ID generation)
        
    Returns:
        Dict containing normalized data
    """
    # Extract politician name and generate ID
    name = data.get("name", "Unknown")
    politician_id = data.get("id", name.lower().replace(" ", "-"))
    
    # Create a normalized structure
    normalized = {
        "id": politician_id,
        "name": name,
        "political_affiliation": data.get("political_affiliation", ""),
        "biography": data.get("raw_content", data.get("biography", "")),
        "date_of_birth": data.get("date_of_birth", ""),
        "source_url": data.get("source_url", ""),
        "policies": "{}"  # Default empty JSON object
    }
    
    # Add other fields if available
    if "policies" in data and isinstance(data["policies"], dict):
        normalized["policies"] = json.dumps(data["policies"])
    elif "policy_positions" in data and isinstance(data["policy_positions"], dict):
        normalized["policies"] = json.dumps(data["policy_positions"])
    
    return normalized

def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load politician data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of politician data dictionaries
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both single object and array formats
        if isinstance(data, list):
            return data
        else:
            return [data]  # Convert single object to a list
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {e}")
        return []

def process_politician_file(file_path: str, collection, stats: Dict[str, int]) -> None:
    """
    Process a single politician data file and load into ChromaDB.
    
    Args:
        file_path: Path to the JSON file
        collection: ChromaDB collection
        stats: Statistics dictionary to update
    """
    try:
        filename = os.path.basename(file_path)
        logger.info(f"Processing file: {filename}")
        
        # Load the JSON data
        politicians_data = load_json_file(file_path)
        
        if not politicians_data:
            logger.warning(f"No data found in {filename}")
            stats["empty_files"] += 1
            return
        
        # Track how many entries we process from this file
        entries_processed = 0
        entries_loaded = 0
        
        # Process each entry in the file
        for data in politicians_data:
            entries_processed += 1
            
            # Normalize the data
            normalized_data = normalize_politician_data(data, filename)
            
            # Skip entries with empty biographies
            if not normalized_data["biography"]:
                logger.warning(f"Skipping entry with empty biography: {normalized_data['name']}")
                stats["empty_entries"] += 1
                continue
                
            # Skip entries that are error pages
            if "page can't be found" in normalized_data["biography"] or "Sorry, we couldn't find that page" in normalized_data["biography"]:
                logger.warning(f"Skipping error page: {normalized_data['name']}")
                stats["error_entries"] += 1
                continue
            
            # Load into ChromaDB
            doc_id = upsert_politician(collection, normalized_data)
            
            if doc_id:
                logger.info(f"Successfully loaded {normalized_data['name']} with ID {doc_id}")
                entries_loaded += 1
                stats["loaded"] += 1
            else:
                logger.error(f"Failed to load {normalized_data['name']}")
                stats["failed"] += 1
        
        logger.info(f"Processed {entries_processed} entries from {filename}, loaded {entries_loaded}")
        stats["processed_files"] += 1
        stats["total_entries"] += entries_processed
    
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        stats["failed_files"] += 1

def process_directory(source_dir: str, db_path: str = DEFAULT_DB_PATH) -> Dict[str, int]:
    """
    Process all JSON files in a directory and load into ChromaDB.
    
    Args:
        source_dir: Directory containing JSON files
        db_path: Path to the ChromaDB database
        
    Returns:
        Dict with statistics about the loading process
    """
    # Initialize statistics
    stats = {
        "processed_files": 0,
        "failed_files": 0,
        "empty_files": 0,
        "total_entries": 0,
        "loaded": 0,
        "failed": 0,
        "empty_entries": 0,
        "error_entries": 0
    }
    
    try:
        # Connect to ChromaDB
        logger.info(f"Connecting to ChromaDB at {db_path}...")
        client = connect_to_chroma(db_path)
        if not client:
            logger.error("Failed to connect to ChromaDB")
            return stats
        
        # Get collection
        collection = get_collection(client)
        if not collection:
            logger.error("Failed to get collection")
            return stats
        
        # Process all JSON files in the directory
        for root, _, files in os.walk(source_dir):
            json_files = [f for f in files if f.endswith('.json')]
            
            if not json_files:
                logger.warning(f"No JSON files found in {source_dir}")
                return stats
            
            logger.info(f"Found {len(json_files)} JSON files in {source_dir}")
            
            for filename in json_files:
                file_path = os.path.join(root, filename)
                process_politician_file(file_path, collection, stats)
        
        return stats
    
    except Exception as e:
        logger.error(f"Error processing directory {source_dir}: {e}")
        return stats

def load_database(source_dir: str, db_path: str = DEFAULT_DB_PATH, verbose: bool = False) -> Dict[str, int]:
    """
    Main function for loading the database when called from another module.
    
    Args:
        source_dir: Directory containing JSON files
        db_path: Path to the ChromaDB database
        verbose: Whether to enable verbose logging
        
    Returns:
        Dict with statistics about the loading process
    """
    # Set logging level
    if verbose:
        logger.setLevel(logging.DEBUG)
    
    # Check if source directory exists
    if not os.path.exists(source_dir):
        logger.error(f"Source directory not found: {source_dir}")
        return {
            "processed_files": 0,
            "failed_files": 0,
            "empty_files": 0,
            "total_entries": 0,
            "loaded": 0,
            "failed": 0,
            "empty_entries": 0,
            "error_entries": 0,
            "error": f"Source directory not found: {source_dir}"
        }
    
    # Process all JSON files in the source directory
    logger.info(f"Processing files from {source_dir}")
    stats = process_directory(source_dir, db_path)
    
    # Print statistics
    logger.info("--- Loading Statistics ---")
    logger.info(f"Processed files: {stats['processed_files']}")
    logger.info(f"Failed files: {stats['failed_files']}")
    logger.info(f"Empty files: {stats['empty_files']}")
    logger.info(f"Total entries found: {stats['total_entries']}")
    logger.info(f"Entries loaded: {stats['loaded']}")
    logger.info(f"Failed entries: {stats['failed']}")
    logger.info(f"Empty entries skipped: {stats['empty_entries']}")
    logger.info(f"Error page entries skipped: {stats['error_entries']}")
    
    return stats

def main():
    """Main function handling command-line arguments."""
    parser = argparse.ArgumentParser(description="Load politician data into ChromaDB")
    
    parser.add_argument("--source-dir", required=True, 
                        help="Directory containing JSON files with politician data")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH,
                        help="Path to ChromaDB database")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Load the database
    load_database(args.source_dir, args.db_path, args.verbose)
    
if __name__ == "__main__":
    main() 