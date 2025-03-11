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
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Crawl4AI imports
from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig

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

if __name__ == "__main__":
    asyncio.run(test_basic_crawl()) 