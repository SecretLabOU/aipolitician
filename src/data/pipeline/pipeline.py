#!/usr/bin/env python3
"""
Political Figure Data Pipeline

This script creates a pipeline that:
1. Scrapes data about political figures using the politician_scraper
2. Processes the data into the correct format for ChromaDB
3. Inserts the data into the ChromaDB vector database
4. Provides progress and error reporting

Usage:
    python3 pipeline.py --politicians "Politician1,Politician2,..."
    python3 pipeline.py --file politicians.txt
"""

import os
import sys
import json
import argparse
import asyncio
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add the scraper directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the scraper and ChromaDB modules
import scraper.politician_scraper as scraper
from scraper.politician_scraper import crawl_political_figure
from db.chroma.schema import connect_to_chroma, get_collection, DEFAULT_DB_PATH
from db.chroma.operations import upsert_politician
import chromadb

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("political_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("political_pipeline")

def map_scraper_to_chroma(scraper_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Maps data from the scraper format to the ChromaDB format.
    
    Args:
        scraper_data: The data returned by the politician scraper
        
    Returns:
        Dict containing the data formatted for ChromaDB insertion
    """
    try:
        # Create a new mapping with the required ChromaDB schema fields
        chroma_data = {
            "id": scraper_data.get("id", str(uuid.uuid4())),
            "name": scraper_data.get("name", ""),
            "date_of_birth": scraper_data["basic_info"].get("date_of_birth", ""),
            "nationality": scraper_data["basic_info"].get("nationality", ""),
            "political_affiliation": scraper_data["basic_info"].get("political_affiliation", ""),
            "biography": "", # We'll construct the biography from various fields
            "positions": "", # We'll construct the positions from various fields
            "policies": json.dumps({
                "economy": scraper_data["policy_positions"].get("economy", []),
                "foreign_policy": scraper_data["policy_positions"].get("foreign_policy", []),
                "healthcare": scraper_data["policy_positions"].get("healthcare", []),
                "immigration": scraper_data["policy_positions"].get("immigration", []),
                "environment": scraper_data["policy_positions"].get("environment", []),
                "social_issues": scraper_data["policy_positions"].get("social_issues", []),
                "other": scraper_data["policy_positions"].get("other", [])
            }),
            "legislative_actions": json.dumps({
                "sponsored_bills": scraper_data["legislative_record"].get("sponsored_bills", []),
                "voting_record": scraper_data["legislative_record"].get("voting_record", []),
                "achievements": scraper_data["legislative_record"].get("achievements", [])
            }),
            "public_communications": json.dumps({
                "speeches": scraper_data["communications"].get("speeches", []),
                "statements": scraper_data["communications"].get("statements", []),
                "publications": scraper_data["communications"].get("publications", [])
            }),
            "timeline": json.dumps(scraper_data.get("timeline", [])),
            "campaigns": json.dumps({
                "elections": scraper_data["campaigns"].get("elections", []),
                "platforms": scraper_data["campaigns"].get("platforms", []),
                "fundraising": scraper_data["campaigns"].get("fundraising", [])
            }),
            "media": json.dumps([]), # Initialize as empty array
            "philanthropy": json.dumps([]), # Initialize as empty array
            "personal_details": json.dumps(scraper_data["basic_info"].get("family", []))
        }
        
        # Construct a comprehensive biography from available fields
        biography_parts = [
            f"{chroma_data['name']} is a {chroma_data['nationality']} political figure",
            f"affiliated with the {chroma_data['political_affiliation']}." if chroma_data['political_affiliation'] else "."
        ]
        
        # Add education if available
        if scraper_data["basic_info"].get("education"):
            education_str = ", ".join(scraper_data["basic_info"]["education"][:3])  # Take up to 3 education entries
            biography_parts.append(f"Education: {education_str}.")
        
        # Add career highlights
        if "career" in scraper_data and isinstance(scraper_data["career"], dict):
            positions_str = ", ".join(str(pos) for pos in scraper_data["career"]["positions"][:3])  # Take up to 3 positions
            biography_parts.append(f"Notable positions: {positions_str}.")
            chroma_data["positions"] = json.dumps(scraper_data["career"].get("positions", []))
        else:
            chroma_data["positions"] = "[]"  # Default empty JSON array
            
        # Create the final biography
        chroma_data["biography"] = " ".join(biography_parts)
        
        return chroma_data
        
    except Exception as e:
        logger.error(f"Error mapping scraper data to ChromaDB format: {e}")
        raise

async def scrape_politician_data(name: str) -> Dict[str, Any]:
    """
    Scrape data for a political figure directly using the crawl_political_figure function.
    """
    try:
        logger.info(f"Scraping data for {name}...")
        # Call the scraper's crawl_political_figure function directly
        data = await crawl_political_figure(name)
        logger.info(f"Successfully scraped data for {name}")
        return data
    except Exception as e:
        logger.error(f"Error scraping data for {name}: {e}")
        return None

async def process_politician(politician_name: str, collection: chromadb.Collection) -> bool:
    """
    Process a single politician: scrape data and insert into ChromaDB.
    
    Args:
        politician_name: Name of the politician to process
        collection: ChromaDB collection to insert data into
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Processing politician: {politician_name}")
        
        # Step 1: Scrape data
        logger.info(f"Scraping data for {politician_name}...")
        scraper_data = await scrape_politician_data(politician_name)
        
        if not scraper_data:
            logger.error(f"Failed to scrape data for {politician_name}")
            return False
            
        logger.info(f"Successfully scraped data for {politician_name}")
        
        # Step 2: Map data to ChromaDB format
        logger.info(f"Mapping data for {politician_name} to ChromaDB format...")
        chroma_data = map_scraper_to_chroma(scraper_data)
        
        # Step 3: Insert into ChromaDB (upsert operation will update if exists)
        logger.info(f"Inserting {politician_name} into ChromaDB...")
        try:
            # Upsert the politician
            doc_id = upsert_politician(collection, chroma_data)
            
            if doc_id:
                logger.info(f"Successfully inserted {politician_name} into ChromaDB with ID {doc_id}")
                return True
            else:
                logger.error(f"Failed to insert {politician_name} into ChromaDB")
                return False
                
        except Exception as e:
            logger.error(f"Error inserting {politician_name} into ChromaDB: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Error processing politician {politician_name}: {e}")
        return False

async def run_pipeline(politicians: List[str], db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """
    Run the full pipeline for a list of politicians.
    
    Args:
        politicians: List of politician names to process
        db_path: Path to the ChromaDB database
        
    Returns:
        Dict with statistics about the pipeline run
    """
    stats = {
        "total": len(politicians),
        "successful": 0,
        "failed": 0,
        "skipped": 0,
        "start_time": datetime.now(),
        "end_time": None,
        "details": {}
    }
    
    try:
        # Connect to ChromaDB
        logger.info("Connecting to ChromaDB database...")
        client = connect_to_chroma(db_path=db_path)
        if not client:
            logger.error("Failed to connect to ChromaDB database")
            raise ConnectionError("Could not connect to ChromaDB database")
            
        # Get or create collection
        logger.info("Getting political_figures collection...")
        collection = get_collection(client)
        if not collection:
            logger.error("Failed to get political_figures collection")
            raise ValueError("Could not get political_figures collection")
        
        # Process each politician
        logger.info(f"Starting processing of {len(politicians)} politicians...")
        for politician in politicians:
            politician = politician.strip()
            if not politician:
                continue
                
            try:
                logger.info(f"Processing: {politician}")
                success = await process_politician(politician, collection)
                
                if success:
                    stats["successful"] += 1
                    stats["details"][politician] = "success"
                    logger.info(f"Successfully processed {politician}")
                else:
                    stats["failed"] += 1
                    stats["details"][politician] = "failed"
                    logger.error(f"Failed to process {politician}")
                    
            except Exception as e:
                stats["failed"] += 1
                stats["details"][politician] = f"error: {str(e)}"
                logger.error(f"Exception processing {politician}: {e}")
                
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
    finally:
        stats["end_time"] = datetime.now()
        duration = stats["end_time"] - stats["start_time"]
        stats["duration_seconds"] = duration.total_seconds()
        
        # Log summary
        logger.info(f"Pipeline completed in {duration}")
        logger.info(f"Total politicians: {stats['total']}")
        logger.info(f"Successful: {stats['successful']}")
        logger.info(f"Failed: {stats['failed']}")
        logger.info(f"Skipped: {stats['skipped']}")
        
        return stats

def read_politicians_from_file(file_path: str) -> List[str]:
    """
    Read a list of politicians from a file, one name per line.
    
    Args:
        file_path: Path to the file containing politician names
        
    Returns:
        List of politician names
    """
    try:
        with open(file_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Error reading politicians from file {file_path}: {e}")
        return []

def save_stats(stats: Dict[str, Any], output_path: Optional[str] = None) -> None:
    """
    Save pipeline statistics to a JSON file.
    
    Args:
        stats: Pipeline statistics
        output_path: Path to save the stats (optional)
    """
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"pipeline_stats_{timestamp}.json"
        
    try:
        # Convert datetime objects to strings
        serializable_stats = stats.copy()
        serializable_stats["start_time"] = stats["start_time"].isoformat()
        serializable_stats["end_time"] = stats["end_time"].isoformat()
        
        with open(output_path, 'w') as f:
            json.dump(serializable_stats, f, indent=2)
            
        logger.info(f"Pipeline statistics saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving pipeline statistics: {e}")

def main():
    """Main entry point for the pipeline script."""
    parser = argparse.ArgumentParser(description="Political Figure Data Pipeline")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--politicians", help="Comma-separated list of politicians to process")
    group.add_argument("--file", help="Path to file containing politicians (one per line)")
    
    parser.add_argument("--stats-output", help="Path to save pipeline statistics (JSON)")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help=f"Path to ChromaDB database (default: {DEFAULT_DB_PATH})")
    
    args = parser.parse_args()
    
    # Get list of politicians
    if args.politicians:
        politicians = [name.strip() for name in args.politicians.split(",")]
    elif args.file:
        politicians = read_politicians_from_file(args.file)
    else:
        parser.error("Either --politicians or --file must be provided")
    
    if not politicians:
        logger.error("No politicians specified")
        sys.exit(1)
        
    logger.info(f"Starting pipeline for {len(politicians)} politicians")
    
    # Run the pipeline
    stats = asyncio.run(run_pipeline(politicians, args.db_path))
    
    # Save statistics
    save_stats(stats, args.stats_output)
    
    # Report results
    success_rate = (stats["successful"] / stats["total"]) * 100 if stats["total"] > 0 else 0
    if stats["failed"] > 0:
        logger.warning(f"Pipeline completed with {stats['failed']} failures ({success_rate:.1f}% success rate)")
        sys.exit(1)
    else:
        logger.info(f"Pipeline completed successfully ({success_rate:.1f}% success rate)")
        sys.exit(0)

if __name__ == "__main__":
    main()