#!/usr/bin/env python3
"""
Test Script for Politician Data Scraper

This script tests the politician data scraper without inserting data into the Milvus database.
It allows you to verify that the scraper is working correctly before committing data to the database.

Usage:
    python test_politician_scraper.py --name "Politician Name" [--depth 3] [--max_pages 20]

Example:
    python test_politician_scraper.py --name "Donald Trump"
    python test_politician_scraper.py --name "Joe Biden" --depth 4 --max_pages 30
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to the Python path
root_dir = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(root_dir))

# Create logs directory if it doesn't exist
logs_dir = os.path.join(root_dir, "scraper", "logs")
os.makedirs(logs_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import the MockWebCrawler from the politician_scraper module
try:
    from scraper.politician_scraper import MockWebCrawler
except ImportError:
    # If the module doesn't exist yet, define a simple version here
    from typing import Dict, List, Any
    import json
    
    class MockWebCrawler:
        """Mock implementation of the WebCrawler for testing purposes"""
        
        def __init__(self, depth=3, max_pages=20):
            self.depth = depth
            self.max_pages = max_pages
            print("Using mock web crawler for testing.")
        
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

# Try to import the real Crawl4AI crawler if available
try:
    import requests
    HAS_CRAWL4AI = True
except ImportError:
    HAS_CRAWL4AI = False
    print("Requests library not installed. Using mock crawler only.")

class TestPoliticianScraper:
    """
    Test scraper for politician data that uses either a mock crawler or the Crawl4AI API
    to gather information and processes it without storing in the database.
    """
    
    def __init__(self, depth: int = 3, max_pages: int = 20, use_crawl4ai: bool = False):
        """
        Initialize the test scraper.
        
        Args:
            depth (int): Maximum depth for web crawling
            max_pages (int): Maximum number of pages to crawl
            use_crawl4ai (bool): Whether to use the Crawl4AI API instead of the mock crawler
        """
        self.depth = depth
        self.max_pages = max_pages
        self.use_crawl4ai = use_crawl4ai and HAS_CRAWL4AI
        
        # Initialize web crawler
        if not self.use_crawl4ai:
            self.crawler = MockWebCrawler(depth=depth, max_pages=max_pages)
            logger.info("Using mock web crawler for testing")
        else:
            logger.info("Using Crawl4AI API for testing")
    
    def test_scrape_politician(self, name: str) -> Dict[str, Any]:
        """
        Test scraping data about a politician without storing in the database.
        
        Args:
            name (str): Name of the politician to scrape
            
        Returns:
            dict: Structured data about the politician
        """
        logger.info(f"Test scraping data for politician: {name}")
        
        try:
            # Construct search query
            search_query = f"{name} politician biography policies positions"
            
            # Get data either from mock crawler or Crawl4AI
            if not self.use_crawl4ai:
                # Use mock crawler
                results = self.crawler.crawl(search_query)
            else:
                # Use Crawl4AI API
                results = self._crawl_with_crawl4ai(name)
            
            if not results:
                logger.warning(f"No results found for {name}")
                return None
            
            # Process and clean the data
            politician_data = self._process_crawl_results(name, results)
            
            return politician_data
            
        except Exception as e:
            logger.error(f"Error scraping data for {name}: {str(e)}")
            raise
    
    def _crawl_with_crawl4ai(self, name: str) -> List[Dict[str, Any]]:
        """
        Use the Crawl4AI API to crawl for politician data.
        
        Args:
            name (str): Name of the politician to scrape
            
        Returns:
            list: List of crawl results
        """
        try:
            # Check if Crawl4AI is running
            try:
                health = requests.get("http://localhost:11235/health", timeout=5)
                logger.info(f"Crawl4AI health check: {health.json()}")
            except requests.exceptions.ConnectionError:
                logger.error("Could not connect to Crawl4AI. Is it running?")
                logger.error("Try running: docker run -p 11235:11235 unclecode/crawl4ai:latest")
                return []
            except requests.exceptions.JSONDecodeError:
                logger.error("Crawl4AI returned invalid JSON. There might be a platform compatibility issue.")
                logger.error("Try running: docker run --platform linux/amd64 -p 11235:11235 unclecode/crawl4ai:latest")
                return []
            
            # Define search URLs based on the politician name
            urls = [
                f"https://en.wikipedia.org/wiki/{name.replace(' ', '_')}",
                f"https://www.britannica.com/biography/{name.replace(' ', '-')}",
                f"https://ballotpedia.org/{name.replace(' ', '_')}"
            ]
            
            # Use LLM extraction for detailed information
            request = {
                "urls": urls,
                "extraction_config": {
                    "type": "cosine",  # Use cosine similarity instead of LLM which requires API keys
                    "params": {
                        "semantic_filter": f"{name} politician biography policies positions",
                        "word_count_threshold": 10,
                        "max_dist": 0.3,
                        "top_k": 10
                    }
                }
            }
            
            # Submit crawl request
            response = requests.post(
                "http://localhost:11235/crawl",
                json=request,
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"Crawl4AI API error: {response.text}")
                return []
            
            task_id = response.json()["task_id"]
            logger.info(f"Crawl4AI task ID: {task_id}")
            
            # Wait for results (in a real implementation, you might want to poll)
            logger.info("Waiting for Crawl4AI results (this may take a while)...")
            result_response = requests.get(f"http://localhost:11235/task/{task_id}", timeout=300)
            
            if result_response.status_code != 200:
                logger.error(f"Crawl4AI result error: {result_response.text}")
                return []
            
            result_data = result_response.json()
            
            # Format results to match the expected structure
            formatted_results = []
            for url_result in result_data.get("results", []):
                formatted_results.append({
                    "url": url_result.get("url", ""),
                    "title": url_result.get("title", ""),
                    "content": url_result.get("markdown", ""),
                    "metadata": self._extract_metadata_from_content(url_result.get("markdown", ""), name)
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error using Crawl4AI: {str(e)}")
            return []
    
    def _extract_metadata_from_content(self, content: str, name: str) -> Dict[str, Any]:
        """
        Extract metadata from the content using simple heuristics.
        In a real implementation, this would be more sophisticated.
        
        Args:
            content (str): The content to extract metadata from
            name (str): The name of the politician
            
        Returns:
            dict: Extracted metadata
        """
        metadata = {
            "date_of_birth": "",
            "nationality": "",
            "political_affiliation": "",
            "positions": {},
            "policies": {}
        }
        
        # Very simple extraction based on keywords
        lines = content.split('\n')
        for line in lines:
            line = line.lower()
            
            if "born" in line and "19" in line:  # Simple date detection
                # Try to extract date in format YYYY-MM-DD
                import re
                date_match = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', line)
                if date_match:
                    metadata["date_of_birth"] = date_match.group(0)
                else:
                    # Try to extract year
                    year_match = re.search(r'19\d{2}', line)
                    if year_match:
                        metadata["date_of_birth"] = year_match.group(0) + "-01-01"
            
            if "american" in line:
                metadata["nationality"] = "American"
            elif "british" in line:
                metadata["nationality"] = "British"
            
            if "republican" in line:
                metadata["political_affiliation"] = "Republican Party"
            elif "democrat" in line:
                metadata["political_affiliation"] = "Democratic Party"
            
            # Extract positions
            if "president" in line:
                metadata["positions"]["President"] = {"details": line}
            elif "senator" in line:
                metadata["positions"]["Senator"] = {"details": line}
            elif "governor" in line:
                metadata["positions"]["Governor"] = {"details": line}
            
            # Extract policies
            if "healthcare" in line or "health care" in line:
                metadata["policies"]["healthcare"] = {"details": line}
            elif "climate" in line or "environment" in line:
                metadata["policies"]["climate"] = {"details": line}
            elif "tax" in line or "economy" in line:
                metadata["policies"]["economy"] = {"details": line}
            elif "immigration" in line:
                metadata["policies"]["immigration"] = {"details": line}
        
        # Convert to JSON strings
        metadata["positions"] = json.dumps(metadata["positions"])
        metadata["policies"] = json.dumps(metadata["policies"])
        
        return metadata
    
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
            "id": "test-id-not-for-db",  # Use a test ID
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
                                if isinstance(politician_data[field], dict):
                                    politician_data[field].update(parsed_data)
                                else:
                                    # If it's already a string, parse it first
                                    try:
                                        existing_data = json.loads(politician_data[field])
                                        existing_data.update(parsed_data)
                                        politician_data[field] = existing_data
                                    except:
                                        politician_data[field] = parsed_data
                            else:
                                politician_data[field] = parsed_data
                        except json.JSONDecodeError:
                            logger.warning(f"Could not parse JSON for field {field}")
        
        # Set biography from combined content
        politician_data["biography"] = combined_content.strip()
        
        # Convert dictionary fields to JSON strings for consistency with the main scraper
        for field in ["positions", "policies", "legislative_actions", "public_communications", 
                     "timeline", "campaigns", "media", "philanthropy", "personal_details"]:
            if politician_data[field] and not isinstance(politician_data[field], str):
                politician_data[field] = json.dumps(politician_data[field])
        
        return politician_data
    
    def print_politician_data(self, politician_data: Dict[str, Any]) -> None:
        """
        Print the politician data in a readable format.
        
        Args:
            politician_data (dict): Structured data about the politician
        """
        if not politician_data:
            print("No data available")
            return
        
        print("\n" + "="*50)
        print(f"TEST RESULTS FOR: {politician_data['name']}")
        print("="*50)
        
        print(f"\nBASIC INFORMATION:")
        print(f"Name: {politician_data['name']}")
        print(f"Date of Birth: {politician_data['date_of_birth']}")
        print(f"Nationality: {politician_data['nationality']}")
        print(f"Political Affiliation: {politician_data['political_affiliation']}")
        
        print(f"\nBIOGRAPHY (excerpt):")
        # Print only the first 200 characters of the biography
        biography = politician_data['biography']
        print(f"{biography[:200]}..." if len(biography) > 200 else biography)
        
        # Print JSON fields in a readable format
        for field in ["positions", "policies", "legislative_actions", "public_communications", 
                     "timeline", "campaigns", "media", "philanthropy", "personal_details"]:
            if politician_data[field] and politician_data[field] != "{}":
                print(f"\n{field.upper()}:")
                try:
                    if isinstance(politician_data[field], str):
                        data = json.loads(politician_data[field])
                    else:
                        data = politician_data[field]
                    
                    # Print the first few items
                    items = list(data.items())[:3]
                    for key, value in items:
                        print(f"  {key}: {value}")
                    
                    if len(data) > 3:
                        print(f"  ... and {len(data) - 3} more items")
                except:
                    print(f"  Could not parse JSON: {politician_data[field][:100]}...")
        
        print("\n" + "="*50)
        print("TEST COMPLETED SUCCESSFULLY - NO DATA WAS STORED IN THE DATABASE")
        print("="*50 + "\n")


def main():
    """Main function to run the test scraper"""
    parser = argparse.ArgumentParser(description="Test scraping data about politicians without storing in the database")
    parser.add_argument("--name", type=str, required=True, help="Name of the politician to scrape")
    parser.add_argument("--depth", type=int, default=3, help="Maximum depth for web crawling")
    parser.add_argument("--max_pages", type=int, default=20, help="Maximum number of pages to crawl")
    parser.add_argument("--use-crawl4ai", action="store_true", help="Use the Crawl4AI API instead of the mock crawler")
    
    args = parser.parse_args()
    
    try:
        # Initialize the test scraper
        scraper = TestPoliticianScraper(
            depth=args.depth, 
            max_pages=args.max_pages,
            use_crawl4ai=args.use_crawl4ai
        )
        
        # Test scrape data for the specified politician
        politician_data = scraper.test_scrape_politician(args.name)
        
        if politician_data:
            # Print the results
            scraper.print_politician_data(politician_data)
            logger.info(f"Successfully tested scraping for {args.name}")
        else:
            logger.error(f"Failed to scrape test data for {args.name}")
            print(f"Failed to scrape test data for {args.name}")
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 