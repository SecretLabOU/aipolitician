#!/usr/bin/env python3
"""
Minimal test script for Crawl4AI

This script provides a minimal test to verify that Crawl4AI is working correctly.
It performs a basic crawl operation without any complex extraction strategies.

Usage:
    python -m scraper.minimal_test
"""

import asyncio
import logging
import sys
import os
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Crawl4AI imports
from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy

# Try to import LlmConfig from different possible locations
try:
    from crawl4ai import LlmConfig
    print("Imported LlmConfig from crawl4ai")
except ImportError:
    try:
        from crawl4ai.config import LlmConfig
        print("Imported LlmConfig from crawl4ai.config")
    except ImportError:
        try:
            from crawl4ai.llm import LlmConfig
            print("Imported LlmConfig from crawl4ai.llm")
        except ImportError:
            # Define a fallback LlmConfig
            print("Could not import LlmConfig. Using fallback version.")
            class LlmConfig:
                def __init__(self, provider=None, api_token=None):
                    self.provider = provider
                    self.api_token = api_token

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def test_basic_crawl():
    """Test basic web crawling with minimal configuration."""
    print("Testing basic web crawling...")
    
    async with AsyncWebCrawler() as crawler:
        try:
            # Simple crawl with minimal configuration
            config = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS
            )
            
            url = "https://en.wikipedia.org/wiki/Donald_Trump"
            print(f"Crawling {url}...")
            
            result = await crawler.arun(
                url=url,
                config=config
            )
            
            if hasattr(result, 'success') and result.success:
                print("Crawl successful!")
                if hasattr(result, 'markdown'):
                    print(f"Extracted {len(result.markdown)} characters of markdown")
                    print(f"Preview: {result.markdown[:200]}...")
                else:
                    print("No markdown attribute found in result")
                    
                # Display available result attributes
                print("\nAvailable result attributes:")
                for attr in dir(result):
                    if not attr.startswith('_'):
                        try:
                            value = getattr(result, attr)
                            if not callable(value):
                                print(f"  {attr}: {type(value)}")
                        except Exception:
                            pass
            else:
                error_msg = getattr(result, 'error_message', 'Unknown error')
                print(f"Crawl failed: {error_msg}")
                
        except Exception as e:
            print(f"Error during testing: {str(e)}")
            import traceback
            traceback.print_exc()

async def test_simple_extraction():
    """Test a simpler extraction approach that doesn't rely on LLM."""
    print("\nTesting simple extraction without LLM...")
    
    try:
        # Import a simpler extraction strategy
        from crawl4ai.extraction_strategy import PlainTextExtractionStrategy
        
        # Create a simple extraction strategy
        extraction_strategy = PlainTextExtractionStrategy()
        
        # Create crawler config
        config = CrawlerRunConfig(
            extraction_strategy=extraction_strategy,
            cache_mode=CacheMode.BYPASS
        )
        
        # Create crawler and run extraction
        async with AsyncWebCrawler() as crawler:
            url = "https://en.wikipedia.org/wiki/Donald_Trump"
            print(f"Crawling and extracting from {url}...")
            
            result = await crawler.arun(
                url=url,
                config=config
            )
            
            if hasattr(result, 'extraction_result') and result.extraction_result:
                print("Extraction successful!")
                print(f"Extraction result type: {type(result.extraction_result)}")
                print(f"Extraction result preview: {str(result.extraction_result)[:500]}...")
            else:
                print("No extraction result found. Using markdown instead.")
                if hasattr(result, 'markdown') and result.markdown:
                    print(f"Markdown preview: {str(result.markdown)[:500]}...")
                
    except Exception as e:
        print(f"Error during simple extraction test: {str(e)}")
        import traceback
        traceback.print_exc()

async def test_manual_extraction():
    """Test manual extraction from markdown content."""
    print("\nTesting manual extraction from markdown...")
    
    try:
        # First get the content
        config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS
        )
        
        async with AsyncWebCrawler() as crawler:
            url = "https://en.wikipedia.org/wiki/Donald_Trump"
            print(f"Crawling {url}...")
            
            result = await crawler.arun(
                url=url,
                config=config
            )
            
            if hasattr(result, 'markdown') and result.markdown:
                print("Crawl successful! Performing manual extraction...")
                
                # Extract basic information using string operations
                markdown_text = str(result.markdown)
                
                # Example: Extract the title
                import re
                title_match = re.search(r'# (.*?)[\n\r]', markdown_text)
                title = title_match.group(1) if title_match else "Unknown"
                
                # Example: Look for birth date
                birth_date_match = re.search(r'born.*?(\w+ \d+, \d{4})', markdown_text)
                birth_date = birth_date_match.group(1) if birth_date_match else "Unknown"
                
                # Example: Look for political party
                party_match = re.search(r'(Republican|Democratic) Party', markdown_text)
                party = party_match.group(0) if party_match else "Unknown"
                
                # Create a structured result
                extracted_data = {
                    "name": title,
                    "birth_date": birth_date,
                    "political_party": party,
                    "biography_summary": markdown_text[:1000] + "..."
                }
                
                print("Manual extraction results:")
                print(json.dumps(extracted_data, indent=2))
            else:
                print("No markdown content found.")
                
    except Exception as e:
        print(f"Error during manual extraction test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the basic crawl test
    asyncio.run(test_basic_crawl())
    
    # Run the extraction tests if requested
    if "--with-llm" in sys.argv:
        asyncio.run(test_simple_extraction())
        asyncio.run(test_manual_extraction()) 