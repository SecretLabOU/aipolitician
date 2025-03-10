#!/usr/bin/env python3
"""
Batch Politician Data Scraper

This script processes a list of politicians, scraping data for each one
and storing it in the Milvus database.

Usage:
    python -m scraper.batch_scrape [--politicians "Name1" "Name2" ...] [--refresh]
"""

import os
import sys
import time
import asyncio
import argparse
import logging
from typing import List
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import from the scraper package
from scraper.politician_scraper import PoliticianScraper

# Configure logging
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOGS_DIR / "batch_scraper.log"), mode="a")
    ]
)
logger = logging.getLogger(__name__)

# Default list of politicians to scrape
DEFAULT_POLITICIANS = [
    "Donald Trump", 
    "Joe Biden"
]

async def process_politician(name: str, refresh: bool = False) -> bool:
    """
    Process a single politician - scrape and save to database.
    
    Args:
        name: Name of the politician
        refresh: Whether to refresh existing data
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Starting to process: {name}")
        
        # Create and run the scraper
        scraper = PoliticianScraper(name, refresh)
        await scraper.scrape()
        
        # Save to Milvus
        success = await scraper.save_to_milvus()
        
        if success:
            logger.info(f"Successfully processed: {name}")
            return True
        else:
            logger.error(f"Failed to save to database: {name}")
            return False
            
    except Exception as e:
        logger.error(f"Error processing {name}: {str(e)}")
        return False

async def batch_process(politicians: List[str], refresh: bool = False) -> None:
    """
    Process a batch of politicians sequentially.
    
    Args:
        politicians: List of politician names to process
        refresh: Whether to refresh existing data
    """
    logger.info(f"Starting batch processing of {len(politicians)} politicians")
    
    successful = 0
    failed = 0
    
    for name in politicians:
        # Add a delay between politicians to avoid rate limiting
        if successful > 0 or failed > 0:
            logger.info("Waiting 30 seconds before processing next politician...")
            time.sleep(30)
        
        # Process the politician
        result = await process_politician(name, refresh)
        
        if result:
            successful += 1
        else:
            failed += 1
    
    logger.info(f"Batch processing completed: {successful} successful, {failed} failed")

async def main():
    """Main function to parse arguments and run the batch processor."""
    parser = argparse.ArgumentParser(description="Batch process politicians and store in Milvus")
    parser.add_argument("--politicians", type=str, nargs="+", help="List of politician names to process")
    parser.add_argument("--refresh", action="store_true", help="Refresh existing data")
    args = parser.parse_args()
    
    # Use provided list or default list
    politicians = args.politicians if args.politicians else DEFAULT_POLITICIANS
    
    await batch_process(politicians, args.refresh)

if __name__ == "__main__":
    asyncio.run(main()) 