#!/usr/bin/env python3
"""
Politician Data Scraper

This script scrapes data about politicians using an open-source LLM Web Crawler,
cleans the data, and stores it in the Milvus database.

Usage:
    python politician_scraper.py --name "Politician Name" [--depth 3] [--max_pages 20]

Example:
    python politician_scraper.py --name "Donald Trump"
    python politician_scraper.py --name "Joe Biden" --depth 4 --max_pages 30
"""

import argparse
import json
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to the Python path
root_dir = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(root_dir))

# Import Milvus database utilities
from db.milvus.scripts.search import insert_political_figure, connect_to_milvus
from db.milvus.scripts.schema import initialize_database
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(root_dir, "scraper", "logs", f"scraper_{datetime.now().strftime('%Y%m%d')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Try to create logs directory if it doesn't exist
try:
    os.makedirs(os.path.join(root_dir, "scraper", "logs"), exist_ok=True)
except Exception as e:
    logger.warning(f"Could not create logs directory: {str(e)}")

# Initialize the LLM Web Crawler
try:
    from llm_web_crawler import WebCrawler
    HAS_CRAWLER = True
except ImportError:
    HAS_CRAWLER = False
    logger.error("LLM Web Crawler not installed. Please install it with: pip install llm-web-crawler")
    logger.error("For this example, we'll use a mock implementation.")

class MockWebCrawler:
    """Mock implementation of the WebCrawler for testing purposes"""
    
    def __init__(self, depth=3, max_pages=20):
        self.depth = depth
        self.max_pages = max_pages
        logger.warning("Using mock web crawler. Install llm-web-crawler for actual scraping.")
    
    def crawl(self, query: str) -> List[Dict[str, Any]]:
        """Mock crawl method that returns sample data"""
        if "trump" in query.lower():
            return self._get_mock_trump_data()
        elif "biden" in query.lower():
            return self._get_mock_biden_data()
        else:
            return self._get_generic_politician_data(query)
    
    def _get_mock_trump_data(self) -> List[Dict[str, Any]]:
        """Return mock data for Donald Trump"""
        return [{
            "url": "https://en.wikipedia.org/wiki/Donald_Trump",
            "title": "Donald Trump - Wikipedia",
            "content": "Donald John Trump (born June 14, 1946) is an American politician, media personality, and businessman who served as the 45th president of the United States from 2017 to 2021.",
            "metadata": {
                "date_of_birth": "1946-06-14",
                "nationality": "American",
                "political_affiliation": "Republican Party",
                "positions": json.dumps({
                    "45th President of the United States": {"start": "2017-01-20", "end": "2021-01-20"},
                    "Chairman of The Trump Organization": {"start": "1971", "end": "2017"}
                }),
                "policies": json.dumps({
                    "immigration": {"position": "restrictive", "details": "Advocated for border wall with Mexico"},
                    "economy": {"position": "pro-business", "details": "Tax cuts and deregulation"}
                })
            }
        }]
    
    def _get_mock_biden_data(self) -> List[Dict[str, Any]]:
        """Return mock data for Joe Biden"""
        return [{
            "url": "https://en.wikipedia.org/wiki/Joe_Biden",
            "title": "Joe Biden - Wikipedia",
            "content": "Joseph Robinette Biden Jr. (born November 20, 1942) is an American politician who is the 46th and current president of the United States. A member of the Democratic Party, he previously served as the 47th vice president from 2009 to 2017 under President Barack Obama.",
            "metadata": {
                "date_of_birth": "1942-11-20",
                "nationality": "American",
                "political_affiliation": "Democratic Party",
                "positions": json.dumps({
                    "46th President of the United States": {"start": "2021-01-20", "end": "present"},
                    "47th Vice President of the United States": {"start": "2009-01-20", "end": "2017-01-20"}
                }),
                "policies": json.dumps({
                    "climate_change": {"position": "supportive", "details": "Rejoined Paris Climate Agreement"},
                    "healthcare": {"position": "supportive", "details": "Expanded Affordable Care Act"}
                })
            }
        }]
    
    def _get_generic_politician_data(self, name: str) -> List[Dict[str, Any]]:
        """Return generic mock data for any politician"""
        return [{
            "url": f"https://en.wikipedia.org/wiki/{name.replace(' ', '_')}",
            "title": f"{name} - Wikipedia",
            "content": f"{name} is a politician.",
            "metadata": {
                "date_of_birth": "",
                "nationality": "",
                "political_affiliation": "",
                "positions": json.dumps({}),
                "policies": json.dumps({})
            }
        }]


class PoliticianScraper:
    """
    Scraper for politician data that uses an LLM Web Crawler to gather information,
    cleans the data, and stores it in the Milvus database.
    """
    
    def __init__(self, depth: int = 3, max_pages: int = 20):
        """
        Initialize the scraper.
        
        Args:
            depth (int): Maximum depth for web crawling
            max_pages (int): Maximum number of pages to crawl
        """
        self.depth = depth
        self.max_pages = max_pages
        
        # Initialize web crawler
        if HAS_CRAWLER:
            self.crawler = WebCrawler(depth=depth, max_pages=max_pages)
        else:
            self.crawler = MockWebCrawler(depth=depth, max_pages=max_pages)
        
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            self.embedding_model = None
        
        # Connect to Milvus database
        try:
            connect_to_milvus()
            initialize_database()
            logger.info("Connected to Milvus database")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus database: {str(e)}")
            raise
    
    def scrape_politician(self, name: str) -> Dict[str, Any]:
        """
        Scrape data about a politician.
        
        Args:
            name (str): Name of the politician to scrape
            
        Returns:
            dict: Structured data about the politician
        """
        logger.info(f"Scraping data for politician: {name}")
        
        try:
            # Construct search query
            search_query = f"{name} politician biography policies positions"
            
            # Crawl the web for information
            results = self.crawler.crawl(search_query)
            
            if not results:
                logger.warning(f"No results found for {name}")
                return None
            
            # Process and clean the data
            politician_data = self._process_crawl_results(name, results)
            
            # Store the data in Milvus
            self._store_in_milvus(politician_data)
            
            return politician_data
            
        except Exception as e:
            logger.error(f"Error scraping data for {name}: {str(e)}")
            raise
    
    def _process_crawl_results(self, name: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process and clean the crawl results.
        
        Args:
            name (str): Name of the politician
            results (list): List of crawl results
            
        Returns:
            dict: Structured data about the politician
        """
        logger.info(f"Processing crawl results for {name}")
        
        # Initialize politician data structure
        politician_data = {
            "id": str(uuid.uuid4()),
            "name": name,
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
        
        # Combine and clean the data from all results
        combined_content = ""
        for result in results:
            combined_content += result.get("content", "") + " "
            
            # Extract metadata if available
            metadata = result.get("metadata", {})
            if metadata:
                # Update fields if they're empty and metadata has values
                if not politician_data["date_of_birth"] and metadata.get("date_of_birth"):
                    politician_data["date_of_birth"] = metadata["date_of_birth"]
                
                if not politician_data["nationality"] and metadata.get("nationality"):
                    politician_data["nationality"] = metadata["nationality"]
                
                if not politician_data["political_affiliation"] and metadata.get("political_affiliation"):
                    politician_data["political_affiliation"] = metadata["political_affiliation"]
                
                # Parse JSON fields if available
                for field in ["positions", "policies", "legislative_actions", "public_communications", 
                             "timeline", "campaigns", "media", "philanthropy", "personal_details"]:
                    if metadata.get(field):
                        try:
                            if isinstance(metadata[field], str):
                                parsed_data = json.loads(metadata[field])
                            else:
                                parsed_data = metadata[field]
                            
                            # Merge with existing data
                            if politician_data[field]:
                                politician_data[field].update(parsed_data)
                            else:
                                politician_data[field] = parsed_data
                        except json.JSONDecodeError:
                            logger.warning(f"Could not parse JSON for field {field}")
        
        # Set biography from combined content
        politician_data["biography"] = combined_content.strip()
        
        # Convert dictionary fields to JSON strings for Milvus storage
        for field in ["positions", "policies", "legislative_actions", "public_communications", 
                     "timeline", "campaigns", "media", "philanthropy", "personal_details"]:
            if politician_data[field] and not isinstance(politician_data[field], str):
                politician_data[field] = json.dumps(politician_data[field])
        
        return politician_data
    
    def _store_in_milvus(self, politician_data: Dict[str, Any]) -> None:
        """
        Store the politician data in the Milvus database.
        
        Args:
            politician_data (dict): Structured data about the politician
        """
        logger.info(f"Storing data for {politician_data['name']} in Milvus")
        
        try:
            # Insert the politician data into Milvus
            insert_political_figure(politician_data)
            logger.info(f"Successfully stored data for {politician_data['name']} in Milvus")
        except Exception as e:
            logger.error(f"Error storing data in Milvus: {str(e)}")
            raise


def main():
    """Main function to run the scraper"""
    parser = argparse.ArgumentParser(description="Scrape data about politicians")
    parser.add_argument("--name", type=str, required=True, help="Name of the politician to scrape")
    parser.add_argument("--depth", type=int, default=3, help="Maximum depth for web crawling")
    parser.add_argument("--max_pages", type=int, default=20, help="Maximum number of pages to crawl")
    
    args = parser.parse_args()
    
    try:
        # Initialize the scraper
        scraper = PoliticianScraper(depth=args.depth, max_pages=args.max_pages)
        
        # Scrape data for the specified politician
        politician_data = scraper.scrape_politician(args.name)
        
        if politician_data:
            logger.info(f"Successfully scraped and stored data for {args.name}")
            print(f"Successfully scraped and stored data for {args.name}")
        else:
            logger.error(f"Failed to scrape data for {args.name}")
            print(f"Failed to scrape data for {args.name}")
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
