#!/usr/bin/env python3
"""
Politician Data Scraper and Database Filler

This script scrapes data about politicians using crawl4ai, preprocesses the data,
and fills the Milvus database with the processed information.

Usage:
    python -m scraper.politician_scraper --name "Politician Name" [--refresh]
"""

import os
import sys
import json
import uuid
import logging
import argparse
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# Crawl4AI imports - Using simplified imports
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from crawl4ai.config import ContentSelectionConfig

# Add the project root to path to import from db module
# Get the project root directory (parent of the scraper directory)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from db.milvus.scripts.search import insert_political_figure, connect_to_milvus
from db.milvus.scripts.schema import initialize_database

# Configure logging
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOGS_DIR / "politician_scraper.log"), mode="a")
    ]
)
logger = logging.getLogger(__name__)

# Define search queries for different politicians
POLITICIAN_SEARCH_QUERIES = {
    "Donald Trump": [
        "Donald Trump biography",
        "Donald Trump political career",
        "Donald Trump policies",
        "Donald Trump presidency",
        "Donald Trump campaign",
        "Donald Trump recent statements"
    ],
    "Joe Biden": [
        "Joe Biden biography",
        "Joe Biden political career",
        "Joe Biden policies",
        "Joe Biden presidency",
        "Joe Biden campaign",
        "Joe Biden recent statements"
    ]
}

# Default search queries for politicians not in the predefined list
DEFAULT_SEARCH_QUERIES = [
    "{name} biography",
    "{name} political career",
    "{name} policies",
    "{name} recent statements",
    "{name} political positions"
]

class PoliticianScraper:
    """Class to handle scraping of politician data and loading it into Milvus."""
    
    def __init__(self, politician_name: str, refresh: bool = False):
        """
        Initialize the scraper with a politician name.
        
        Args:
            politician_name: Name of the politician to scrape data for
            refresh: Whether to refresh existing data
        """
        self.politician_name = politician_name
        self.refresh = refresh
        self.data = {
            "id": str(uuid.uuid4()),
            "name": politician_name,
            "date_of_birth": "",
            "nationality": "",
            "political_affiliation": "",
            "biography": "",
            "positions": {},
            "policies": {},
            "legislative_actions": {},
            "public_communications": {},
            "timeline": {},
            "campaigns": {},
            "media": {},
            "philanthropy": {},
            "personal_details": {}
        }
        self.search_queries = self._get_search_queries(politician_name)
        
    def _get_search_queries(self, name: str) -> List[str]:
        """Get the search queries for a politician."""
        if name in POLITICIAN_SEARCH_QUERIES:
            return POLITICIAN_SEARCH_QUERIES[name]
        else:
            # For other politicians, use the default template
            return [query.format(name=name) for query in DEFAULT_SEARCH_QUERIES]
    
    async def _extract_structured_data(self, crawler: AsyncWebCrawler, url: str) -> Dict[str, Any]:
        """
        Extract structured data about a politician using LLM strategy
        
        Args:
            crawler: AsyncWebCrawler instance
            url: URL to extract data from
            
        Returns:
            Dict containing structured data
        """
        # Define the extraction prompt
        extraction_prompt = f"""
        Extract information about {self.politician_name} from the content.
        Please provide this information in a structured format:
        
        - date_of_birth: YYYY-MM-DD format if available, otherwise leave empty
        - nationality: Country of origin
        - political_affiliation: Political party or alignment
        - biography_summary: A brief summary of their life and career (up to 300 words)
        - key_positions: Up to 5 political positions held (title and years)
        - policy_stances: Their stance on at least 3 major policy areas
        - recent_statements: Up to 3 recent notable public statements
        
        Return the information in JSON format.
        """
        
        try:
            # Use a simpler approach: just get the markdown content and process it with LLM
            content_config = CrawlerRunConfig(
                content_selection=ContentSelectionConfig(
                    main_content_only=True
                ),
                cache_mode=CacheMode.BYPASS
            )
            
            # Get the content
            result = await crawler.arun(
                url=url,
                config=content_config
            )
            
            if not result.markdown:
                return {}
            
            # Configure the LLM extraction strategy
            llm_strategy = LLMExtractionStrategy(
                provider="local/gpt4all-j" if os.getenv('OPENAI_API_KEY') is None else "openai/gpt-3.5-turbo",
                api_token=os.getenv('OPENAI_API_KEY'),
                instruction=extraction_prompt,
                extraction_type="json",
                input_content=result.markdown[:12000],  # Limit content size to prevent token overflow
                input_format="markdown"
            )
            
            # Create a new config with the LLM strategy
            llm_config = CrawlerRunConfig(
                extraction_strategy=llm_strategy,
                cache_mode=CacheMode.BYPASS
            )
            
            # Run the extraction
            extraction_result = await crawler.arun(
                url="about:blank",  # Use a dummy URL since we're providing the content directly
                config=llm_config
            )
            
            # Parse the extracted JSON result
            if extraction_result.extracted_content:
                try:
                    if isinstance(extraction_result.extracted_content, str):
                        return json.loads(extraction_result.extracted_content)
                    return extraction_result.extracted_content
                except (json.JSONDecodeError, TypeError) as e:
                    logger.error(f"Error parsing JSON from extraction: {e}")
                    return {}
            return {}
        except Exception as e:
            logger.error(f"Error during structured data extraction: {e}")
            return {}
    
    async def scrape(self) -> Dict[str, Any]:
        """
        Scrape data about the politician and return processed data.
        
        Returns:
            Dict containing the processed politician data
        """
        logger.info(f"Starting to scrape data for {self.politician_name}")
        
        try:
            # Set up browser configuration
            browser_config = BrowserConfig(
                headless=True,
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
            )
            
            # Create web crawler
            async with AsyncWebCrawler(config=browser_config) as crawler:
                all_structured_data = []
                
                # We'll use a set of predefined URLs for each politician
                if self.politician_name == "Donald Trump":
                    links = [
                        "https://en.wikipedia.org/wiki/Donald_Trump",
                        "https://www.britannica.com/biography/Donald-Trump",
                        "https://www.whitehouse.gov/about-the-white-house/presidents/donald-j-trump/"
                    ]
                elif self.politician_name == "Joe Biden":
                    links = [
                        "https://en.wikipedia.org/wiki/Joe_Biden",
                        "https://www.britannica.com/biography/Joe-Biden",
                        "https://www.whitehouse.gov/administration/president-biden/"
                    ]
                else:
                    # For other politicians, try to create some reasonable URLs
                    name_slug = self.politician_name.lower().replace(" ", "_")
                    links = [
                        f"https://en.wikipedia.org/wiki/{name_slug}",
                        f"https://www.britannica.com/biography/{name_slug}"
                    ]
                
                # Process each link
                for link in links:
                    logger.info(f"Crawling content from: {link}")
                    
                    try:
                        # Extract structured data
                        structured_data = await self._extract_structured_data(crawler, link)
                        if structured_data:
                            all_structured_data.append(structured_data)
                    
                    except Exception as e:
                        logger.error(f"Error processing link {link}: {str(e)}")
                        continue
                
                # Process and combine all scraped data
                self._process_data(all_structured_data)
                
                return self.data
                
        except Exception as e:
            logger.error(f"Error during scraping: {str(e)}")
            raise
    
    def _process_data(self, all_structured_data: List[Dict[str, Any]]) -> None:
        """
        Process the scraped data and update the data dictionary.
        
        Args:
            all_structured_data: List of structured data extracted via LLM
        """
        logger.info("Processing scraped data")
        
        # Process structured data
        for structured_data in all_structured_data:
            # Update date_of_birth if available and not already set
            if not self.data["date_of_birth"] and structured_data.get("date_of_birth"):
                self.data["date_of_birth"] = structured_data["date_of_birth"]
            
            # Update nationality if available and not already set
            if not self.data["nationality"] and structured_data.get("nationality"):
                self.data["nationality"] = structured_data["nationality"]
            
            # Update political_affiliation if available and not already set
            if not self.data["political_affiliation"] and structured_data.get("political_affiliation"):
                self.data["political_affiliation"] = structured_data["political_affiliation"]
            
            # Append biography summaries
            if structured_data.get("biography_summary"):
                if self.data["biography"]:
                    self.data["biography"] += "\n\n---\n\n" + structured_data["biography_summary"]
                else:
                    self.data["biography"] = structured_data["biography_summary"]
            
            # Collect policy stances
            if structured_data.get("policy_stances"):
                if isinstance(structured_data["policy_stances"], dict):
                    self.data["policies"].update(structured_data["policy_stances"])
                elif isinstance(structured_data["policy_stances"], list):
                    # Convert list to dict if it's a list
                    for i, policy in enumerate(structured_data["policy_stances"]):
                        policy_key = f"policy_{i+1}"
                        self.data["policies"][policy_key] = policy
            
            # Collect key positions
            if structured_data.get("key_positions"):
                if isinstance(structured_data["key_positions"], dict):
                    self.data["positions"].update(structured_data["key_positions"])
                elif isinstance(structured_data["key_positions"], list):
                    for i, position in enumerate(structured_data["key_positions"]):
                        position_key = f"position_{i+1}"
                        self.data["positions"][position_key] = position
            
            # Collect recent statements
            if structured_data.get("recent_statements"):
                statements = structured_data["recent_statements"]
                if isinstance(statements, dict):
                    self.data["public_communications"].update(statements)
                elif isinstance(statements, list):
                    for i, statement in enumerate(statements):
                        statement_key = f"statement_{i+1}"
                        self.data["public_communications"][statement_key] = statement
        
        # Limit biography length
        if self.data["biography"]:
            self.data["biography"] = self.data["biography"][:65000]  # Limit to field size constraint
        
        # Convert dictionary fields to JSON strings for storage
        for field in ["positions", "policies", "legislative_actions", 
                     "public_communications", "timeline", "campaigns", 
                     "media", "philanthropy", "personal_details"]:
            if isinstance(self.data[field], dict) and self.data[field]:
                self.data[field] = json.dumps(self.data[field])
            elif not self.data[field]:
                self.data[field] = json.dumps({})
    
    async def save_to_milvus(self) -> bool:
        """
        Save the processed politician data to Milvus database.
        
        Returns:
            bool: True if save was successful, False otherwise
        """
        logger.info(f"Saving data for {self.politician_name} to Milvus")
        
        try:
            # Connect to Milvus
            if not connect_to_milvus():
                logger.error("Failed to connect to Milvus")
                return False
            
            # Initialize database if needed
            initialize_database()
            
            # Insert the politician data
            insert_political_figure(self.data)
            
            logger.info(f"Successfully saved {self.politician_name} to Milvus")
            return True
        
        except Exception as e:
            logger.error(f"Error saving to Milvus: {str(e)}")
            return False


async def main():
    """Main function to run the script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Scrape data about politicians and store in Milvus")
    parser.add_argument("--name", type=str, required=True, help="Name of the politician to scrape")
    parser.add_argument("--refresh", action="store_true", help="Refresh existing data")
    args = parser.parse_args()
    
    try:
        # Create and run the scraper
        scraper = PoliticianScraper(args.name, args.refresh)
        data = await scraper.scrape()
        
        # Save to Milvus
        success = await scraper.save_to_milvus()
        
        if success:
            logger.info(f"Successfully scraped and stored data for {args.name}")
            return 0
        else:
            logger.error(f"Failed to store data for {args.name}")
            return 1
    
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 