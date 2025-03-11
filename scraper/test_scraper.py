#!/usr/bin/env python3
"""
Test script for the politician scraper

This script is a simplified version of the main scraper that just demonstrates basic functionality
without saving to the database, making it easier to test and debug.

Usage:
    python -m scraper.test_scraper [politician_name]
"""

import asyncio
import json
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import the PoliticianScraper class
from scraper.politician_scraper import PoliticianScraper

async def test_scraper(politician_name: str):
    """Test the scraper for a specific politician."""
    # Create the scraper
    print(f"Initializing scraper for {politician_name}...")
    scraper = PoliticianScraper(politician_name)
    
    # Show search queries
    print("\nSearch queries that will be used:")
    for i, query in enumerate(scraper.search_queries, 1):
        print(f"  {i}. {query}")
    
    # Run the scraper
    print(f"\nStarting scraping process for {politician_name}...")
    start_time = time.time()
    data = await scraper.scrape()
    end_time = time.time()
    
    # Print some summary information from the data
    print(f"\n=== SCRAPING RESULTS ({end_time - start_time:.2f} seconds) ===")
    print(f"Politician: {data['name']}")
    print(f"Date of Birth: {data.get('date_of_birth', 'N/A')}")
    print(f"Political Affiliation: {data.get('political_affiliation', 'N/A')}")
    print(f"Nationality: {data.get('nationality', 'N/A')}")
    
    # Convert JSON string fields back to dictionaries for display
    for field in ["positions", "policies", "public_communications"]:
        if field in data and isinstance(data[field], str):
            try:
                parsed = json.loads(data[field])
                if parsed:
                    print(f"\n{field.title()}:")
                    for key, value in parsed.items():
                        print(f"  - {key}: {value}")
                else:
                    print(f"\n{field.title()}: None found")
            except json.JSONDecodeError:
                print(f"\n{field.title()}: Unable to parse")
    
    # Show a snippet of the biography
    if data.get('biography'):
        print("\nBiography Preview:")
        # Show first 500 characters
        preview = data['biography'][:500]
        print(f"{preview}...")
        print(f"\nTotal biography length: {len(data['biography'])} characters")
    else:
        print("\nBiography: None found")
    
    print("\nTest completed!")

if __name__ == "__main__":
    # Default to Donald Trump if no argument is provided
    politician_name = sys.argv[1] if len(sys.argv) > 1 else "Donald Trump"
    
    # Run the test
    asyncio.run(test_scraper(politician_name)) 