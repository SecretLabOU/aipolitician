#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Politician Data Scraper

A tool to collect and process information about political figures using:
- Scrapy for web crawling
- SpaCy for NER and text processing
- GPU acceleration for processing speed

Usage:
    python politician_scraper.py "Politician Name" [--output-dir DIR] [--env-id ENV_ID] [--gpu-count N]
"""

import os
import sys
import json
import logging
import asyncio
import random
import argparse
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try multiple import approaches to be robust to different execution contexts
try:
    # Try relative import first
    from .politician_crawler.run_crawler import run_spider
except (ImportError, ValueError):
    try:
        # Try absolute import using src path
        from src.data.scraper.politician_crawler.run_crawler import run_spider
    except ImportError:
        # Last resort: modify sys.path and import
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        from politician_crawler.run_crawler import run_spider

async def crawl_political_figure(name: str, max_attempts: int = 3) -> Optional[Dict[str, Any]]:
    """
    Crawl data about a political figure using the Scrapy crawler.
    
    Args:
        name: Name of the political figure to crawl
        max_attempts: Maximum number of attempts to crawl
        
    Returns:
        Dict containing information about the political figure, or None if unsuccessful
    """
    logger.info(f"Crawling data for political figure: {name}")
    
    for attempt in range(1, max_attempts + 1):
        try:
            # Run the spider
            result = await run_spider(name)
            
            # Check if result is valid
            if not result:
                logger.warning(f"Crawler returned no data for {name} (attempt {attempt})")
                continue
                
            # Handle list result (multiple entries per politician)
            if isinstance(result, list):
                if not result:
                    logger.warning(f"Crawler returned empty list for {name} (attempt {attempt})")
                    continue
                    
                # First, look for entries with proper content
                valid_entries = [
                    entry for entry in result 
                    if isinstance(entry, dict) and "name" in entry and entry.get("name") == name
                ]
                
                # If no valid entries found with exact name match, try any with a name field
                if not valid_entries:
                    valid_entries = [
                        entry for entry in result 
                        if isinstance(entry, dict) and "name" in entry
                    ]
                
                if not valid_entries:
                    logger.warning(f"No valid entries found for {name} (attempt {attempt})")
                    continue
                
                # Sort entries by content length to find the most detailed one
                # Prioritize entries with Wikipedia data (these typically have the most info)
                wiki_entries = [e for e in valid_entries if "wikipedia" in str(e.get("source_url", "")).lower()]
                
                if wiki_entries:
                    # Get the entry with the most content from Wikipedia
                    longest_entry = max(
                        wiki_entries, 
                        key=lambda e: len(str(e.get("raw_content", "")))
                    )
                else:
                    # Fall back to entry with the most content
                    longest_entry = max(
                        valid_entries, 
                        key=lambda e: len(str(e.get("raw_content", "")))
                    )
                
                # Use the best entry
                entry = longest_entry
                
                logger.info(f"Successfully crawled data for {name} (attempt {attempt}, selected best of {len(valid_entries)} entries)")
                
                # Add ID to the data if not present
                if "id" not in entry:
                    # Create ID from name (lowercase, replace spaces with dashes)
                    entry["id"] = name.lower().replace(" ", "-")
                
                return entry
                
            # Handle direct dictionary result
            elif isinstance(result, dict) and "name" in result:
                logger.info(f"Successfully crawled data for {name} (attempt {attempt})")
                
                # Add ID to the data if not present
                if "id" not in result:
                    # Create ID from name (lowercase, replace spaces with dashes)
                    result["id"] = name.lower().replace(" ", "-")
                
                return result
            else:
                logger.warning(f"Crawler returned invalid data for {name} (attempt {attempt})")
                
        except Exception as e:
            logger.error(f"Error crawling data for {name} (attempt {attempt}): {e}")
            
        # Wait before retrying
        if attempt < max_attempts:
            wait_time = random.uniform(1.0, 3.0)
            logger.info(f"Waiting {wait_time:.2f}s before retry...")
            await asyncio.sleep(wait_time)
    
    logger.error(f"Failed to crawl data for {name} after {max_attempts} attempts")
    return None

def main():
    """Main function to parse arguments and run the scraper"""
    parser = argparse.ArgumentParser(
        description='Scrape political figure data from the web using Scrapy and SpaCy',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required argument
    parser.add_argument(
        'politician_name',
        type=str,
        help='Name of the political figure to scrape (e.g., "Donald Trump")'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='src/data/scraper/logs',
        help='Directory to save the scraped data JSON files'
    )
    
    parser.add_argument(
        '--env-id',
        type=str,
        default='nat',
        help='GPU environment ID for genv'
    )
    
    parser.add_argument(
        '--gpu-count',
        type=int,
        default=1,
        help='Number of GPUs to use (0 for CPU only)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the spider with provided arguments
    success = asyncio.run(run_spider(
        politician_name=args.politician_name,
        output_dir=args.output_dir,
        env_id=args.env_id,
        gpu_count=args.gpu_count
    ))
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    # Adjust path to allow importing from this package
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(current_dir))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    main()