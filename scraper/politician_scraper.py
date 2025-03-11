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

# Crawl4AI imports
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import LLMExtractionStrategy, JsonCssExtractionStrategy

# Define custom fallback versions right away - will be used if imports fail
class ContentSelectionConfig:
    def __init__(self, main_content_only=True, exclude_selectors=None):
        self.main_content_only = main_content_only
        self.exclude_selectors = exclude_selectors or []

class LlmConfig:
    def __init__(self, provider=None, api_token=None):
        self.provider = provider
        self.api_token = api_token

# Try different possible imports based on crawl4ai version
try:
    # Version where everything is in the main module
    from crawl4ai import ContentSelectionConfig, LlmConfig
    logger = logging.getLogger(__name__)
    logger.info("Using ContentSelectionConfig and LlmConfig from crawl4ai package")
except ImportError:
    try:
        # Version where these might be in config
        from crawl4ai.config import ContentSelectionConfig, LlmConfig
        logger = logging.getLogger(__name__)
        logger.info("Using ContentSelectionConfig and LlmConfig from crawl4ai.config")
    except ImportError:
        try:
            # Version where they might be in separate modules
            from crawl4ai.content import ContentSelectionConfig
            from crawl4ai.llm import LlmConfig
            logger = logging.getLogger(__name__)
            logger.info("Using ContentSelectionConfig from crawl4ai.content and LlmConfig from crawl4ai.llm")
        except ImportError:
            # If all else fails, use our simplified versions for basic functionality
            logger = logging.getLogger(__name__)
            logger.warning("Could not import ContentSelectionConfig and LlmConfig. Using simplified fallback versions.")

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
        Extract structured data about a politician using manual extraction from markdown.
        
        Args:
            crawler: AsyncWebCrawler instance
            url: URL to extract data from
            
        Returns:
            Dict containing structured data
        """
        try:
            # First get the page content with standard settings
            content_config = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS
            )
            
            # Create a custom content selection config
            content_selection = ContentSelectionConfig(
                main_content_only=True,
                exclude_selectors=[
                    "nav", "header", "footer", ".navbar", 
                    ".menu", ".comments", ".ads", "aside"
                ]
            )
            
            # Try to set content selection based on the available API
            try:
                content_config.content_selection = content_selection
            except AttributeError:
                # Fallback for when content_selection attribute doesn't exist
                try:
                    content_config.main_content_only = True
                    content_config.exclude_selectors = content_selection.exclude_selectors
                except AttributeError:
                    pass  # Ignore if we can't set these attributes
            
            # Get the content with flexible parameter handling
            try:
                # Try the modern version first
                result = await crawler.arun(
                    url=url,
                    config=content_config
                )
            except TypeError:
                try:
                    # Try older version with different parameter name
                    result = await crawler.arun(
                        url=url,
                        run_config=content_config
                    )
                except TypeError:
                    # Fallback to simplest call
                    result = await crawler.arun(url=url)
            
            if not hasattr(result, 'markdown') or not result.markdown:
                logger.warning(f"No markdown content extracted from {url}")
                return {}
            
            # Manual extraction from markdown content
            logger.info(f"Performing manual extraction from {url}")
            markdown_text = str(result.markdown)
            
            # Extract structured data using regex patterns
            import re
            
            # Extract date of birth
            dob_patterns = [
                r'born.*?(\w+ \d+,? \d{4})',  # "born June 14, 1946"
                r'born.*?(\d{1,2} \w+ \d{4})',  # "born 14 June 1946"
                r'born.*?(\d{4}-\d{2}-\d{2})',  # "born 1946-06-14"
                r'Birth Date:.*?(\w+ \d+,? \d{4})',  # "Birth Date: June 14, 1946"
                r'Date of Birth:.*?(\w+ \d+,? \d{4})'  # "Date of Birth: June 14, 1946"
            ]
            
            date_of_birth = ""
            for pattern in dob_patterns:
                dob_match = re.search(pattern, markdown_text, re.IGNORECASE)
                if dob_match:
                    date_of_birth = dob_match.group(1)
                    break
            
            # Extract nationality
            nationality_patterns = [
                r'Nationality:.*?(\w+)',  # "Nationality: American"
                r'nationality is (\w+)',  # "nationality is American"
                r'(\w+) (citizen|national)',  # "American citizen"
                r'born in .*?(\w+)'  # "born in America"
            ]
            
            nationality = ""
            for pattern in nationality_patterns:
                nationality_match = re.search(pattern, markdown_text, re.IGNORECASE)
                if nationality_match:
                    nationality = nationality_match.group(1)
                    break
            
            # Extract political affiliation
            party_patterns = [
                r'(Republican|Democratic) Party',  # "Republican Party"
                r'member of the (Republican|Democratic) Party',  # "member of the Republican Party"
                r'affiliated with the (Republican|Democratic) Party',  # "affiliated with the Republican Party"
                r'(Republican|Democrat)',  # "Republican" or "Democrat"
                r'Political Party:.*?(Republican|Democratic)'  # "Political Party: Republican"
            ]
            
            political_affiliation = ""
            for pattern in party_patterns:
                party_match = re.search(pattern, markdown_text, re.IGNORECASE)
                if party_match:
                    political_affiliation = party_match.group(1)
                    break
            
            # Extract biography summary (first 1000 characters)
            biography_summary = markdown_text[:1000] if len(markdown_text) > 0 else ""
            
            # Extract key positions
            position_patterns = [
                r'(\d+)(st|nd|rd|th) president',  # "45th president"
                r'President of the United States',
                r'Vice President',
                r'Secretary of State',
                r'Senator',
                r'Governor',
                r'Mayor',
                r'CEO',
                r'Chairman',
                r'Director'
            ]
            
            positions = {}
            for i, pattern in enumerate(position_patterns):
                position_matches = re.finditer(pattern, markdown_text, re.IGNORECASE)
                for j, match in enumerate(position_matches):
                    position_key = f"position_{i+1}_{j+1}"
                    positions[position_key] = match.group(0)
                    if len(positions) >= 5:  # Limit to 5 positions
                        break
                if len(positions) >= 5:
                    break
            
            # Extract policy stances
            policy_areas = [
                "economy", "healthcare", "immigration", "climate", 
                "education", "foreign policy", "taxes", "gun control",
                "abortion", "environment", "trade", "defense"
            ]
            
            policies = {}
            for i, area in enumerate(policy_areas):
                # Look for sentences containing the policy area
                area_pattern = r'([^.!?]*' + area + r'[^.!?]*[.!?])'
                policy_matches = re.finditer(area_pattern, markdown_text, re.IGNORECASE)
                for j, match in enumerate(policy_matches):
                    if j == 0:  # Take only the first mention of each policy area
                        policies[area] = match.group(1).strip()
                if len(policies) >= 5:  # Limit to 5 policy areas
                    break
            
            # Extract recent statements
            statement_patterns = [
                r'"([^"]{20,})"',  # Text in quotes with at least 20 chars
                r'\'([^\']{20,})\'',  # Text in single quotes with at least 20 chars
                r'said[^.!?]*[,:]?\s*"([^"]+)"',  # "said" followed by quoted text
                r'stated[^.!?]*[,:]?\s*"([^"]+)"'  # "stated" followed by quoted text
            ]
            
            statements = {}
            for i, pattern in enumerate(statement_patterns):
                statement_matches = re.finditer(pattern, markdown_text, re.IGNORECASE)
                for j, match in enumerate(statement_matches):
                    statement_key = f"statement_{i+1}_{j+1}"
                    statements[statement_key] = match.group(1)
                    if len(statements) >= 3:  # Limit to 3 statements
                        break
                if len(statements) >= 3:
                    break
            
            # Combine all extracted data
            structured_data = {
                "date_of_birth": date_of_birth,
                "nationality": nationality,
                "political_affiliation": political_affiliation,
                "biography_summary": biography_summary,
                "key_positions": positions,
                "policy_stances": policies,
                "recent_statements": statements
            }
            
            logger.info(f"Successfully extracted structured data from {url}")
            return structured_data
            
        except Exception as e:
            logger.error(f"Error during structured data extraction: {e}")
            return {}
    
    async def _get_google_search_results(self, crawler: AsyncWebCrawler, query: str, limit: int = 3) -> List[str]:
        """
        Get URLs from Google search results using SimpleCSSSelectorStrategy.
        
        Args:
            crawler: AsyncWebCrawler instance
            query: Search query
            limit: Maximum number of URLs to return
            
        Returns:
            List of URLs from search results
        """
        search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        logger.info(f"Searching Google for: {query}")
        
        try:
            # Define a simpler schema for Google search results that doesn't use baseSelector
            # This avoids the 'baseSelector' error
            try:
                # Try a simpler approach with direct selectors
                link_strategy = JsonCssExtractionStrategy(
                    selectors={
                        "links": ".yuRUbf > a"
                    },
                    attribute_map={
                        "links": {"href": "href"}
                    }
                )
            except (TypeError, ValueError) as e:
                logger.error(f"Error creating JsonCssExtractionStrategy: {e}")
                # Try an even simpler approach as fallback
                try:
                    from crawl4ai.extraction_strategy import PlainTextExtractionStrategy
                    logger.info("Falling back to PlainTextExtractionStrategy for Google search")
                    link_strategy = PlainTextExtractionStrategy()
                except (ImportError, AttributeError):
                    logger.error("Could not create any extraction strategy for Google search")
                    return []
            
            # Create config for search
            search_config = CrawlerRunConfig(
                extraction_strategy=link_strategy,
                cache_mode=CacheMode.BYPASS
            )
            
            # Execute search with flexible parameter handling
            try:
                # Try the modern version first
                search_result = await crawler.arun(
                    url=search_url,
                    config=search_config
                )
            except TypeError:
                try:
                    # Try older version with different parameter name
                    search_result = await crawler.arun(
                        url=search_url,
                        run_config=search_config
                    )
                except TypeError:
                    # Fallback to simplest call
                    search_result = await crawler.arun(
                        url=search_url,
                        extraction_strategy=link_strategy
                    )
            
            # Extract URLs from results - use a more manual approach
            urls = []
            
            # If we have plain text, try to extract URLs using regex
            if hasattr(search_result, 'markdown') and search_result.markdown:
                import re
                # Look for URLs in the markdown
                url_pattern = re.compile(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w\.-]*(?:\?\S+)?')
                found_urls = url_pattern.findall(str(search_result.markdown))
                
                # Filter URLs to exclude Google's own URLs
                urls = [url for url in found_urls if 'google.com' not in url]
                logger.info(f"Extracted {len(urls)} URLs from markdown using regex")
            
            # Also try extraction_result if available
            if hasattr(search_result, 'extraction_result') and search_result.extraction_result:
                result_data = search_result.extraction_result
                
                # Handle different formats of extraction results
                if isinstance(result_data, dict) and "links" in result_data:
                    for link_data in result_data["links"]:
                        if isinstance(link_data, dict) and "href" in link_data:
                            urls.append(link_data["href"])
                elif isinstance(result_data, list):
                    # Some versions might return a list directly
                    for item in result_data:
                        if isinstance(item, dict) and "href" in item:
                            urls.append(item["href"])
                        elif isinstance(item, str) and item.startswith("http"):
                            urls.append(item)
            
            # Limit to requested number and remove duplicates
            urls = list(dict.fromkeys(urls))[:limit]
            
            logger.info(f"Found {len(urls)} URLs from Google search: {urls}")
            return urls
            
        except Exception as e:
            logger.error(f"Error during Google search: {str(e)}")
            return []
    
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
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
                verbose=True  # Help debug
            )
            
            # Create web crawler with flexible parameters to handle different versions
            try:
                # Try the modern version first
                crawler = AsyncWebCrawler(config=browser_config)
            except TypeError:
                try:
                    # Try older version that might use different parameter name
                    crawler = AsyncWebCrawler(browser_config=browser_config)
                except TypeError:
                    # Fallback to simplest initialization
                    crawler = AsyncWebCrawler()
            
            async with crawler:
                all_structured_data = []
                
                # Try to get sources from Google search first
                sources = []
                for query in self.search_queries[:2]:  # Limit to first 2 queries to avoid too many requests
                    urls = await self._get_google_search_results(crawler, query)
                    sources.extend(urls)
                
                # If we couldn't get sources from Google, use predefined URLs
                if not sources:
                    logger.info("No results from Google search, using predefined sources")
                    if self.politician_name == "Donald Trump":
                        sources = [
                            "https://en.wikipedia.org/wiki/Donald_Trump",
                            "https://www.britannica.com/biography/Donald-Trump",
                            "https://www.whitehouse.gov/about-the-white-house/presidents/donald-j-trump/"
                        ]
                    elif self.politician_name == "Joe Biden":
                        sources = [
                            "https://en.wikipedia.org/wiki/Joe_Biden",
                            "https://www.britannica.com/biography/Joe-Biden",
                            "https://www.whitehouse.gov/administration/president-biden/"
                        ]
                    else:
                        # For other politicians, create reasonable URLs
                        name_slug = self.politician_name.lower().replace(" ", "_")
                        sources = [
                            f"https://en.wikipedia.org/wiki/{name_slug}",
                            f"https://www.britannica.com/biography/{name_slug}"
                        ]
                
                # Make sources unique
                sources = list(dict.fromkeys(sources))
                
                # Limit to first 5 sources
                sources = sources[:5]
                logger.info(f"Processing {len(sources)} sources: {sources}")
                
                # Process all sources sequentially
                for url in sources:
                    logger.info(f"Extracting data from: {url}")
                    
                    try:
                        # Process each source to extract structured data
                        structured_data = await self._extract_structured_data(crawler, url)
                        if structured_data:
                            all_structured_data.append(structured_data)
                            logger.info(f"Successfully extracted structured data from {url}")
                        else:
                            logger.warning(f"No structured data extracted from {url}")
                    
                    except Exception as e:
                        logger.error(f"Error processing source {url}: {str(e)}")
                        continue
                
                # Process the extracted data
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
            try:
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
            except Exception as e:
                logger.error(f"Error processing structured data item: {e}")
                continue
        
        # Limit biography length
        if self.data["biography"]:
            self.data["biography"] = self.data["biography"][:65000]  # Limit to field size constraint
        
        # Convert dictionary fields to JSON strings for storage
        for field in ["positions", "policies", "legislative_actions", 
                     "public_communications", "timeline", "campaigns", 
                     "media", "philanthropy", "personal_details"]:
            try:
                if isinstance(self.data[field], dict) and self.data[field]:
                    self.data[field] = json.dumps(self.data[field])
                elif not self.data[field]:
                    self.data[field] = json.dumps({})
            except Exception as e:
                logger.error(f"Error converting {field} to JSON: {e}")
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