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

async def test_llm_extraction():
    """Test LLM extraction with LlmConfig."""
    print("\nTesting LLM extraction with LlmConfig...")
    
    # Set up API token (use environment variable if available)
    api_token = os.getenv('OPENAI_API_KEY')
    provider = "openai/gpt-4o-mini" if api_token else "ollama/llama3"
    
    if not api_token and provider.startswith("openai/"):
        print("No OpenAI API key found. Skipping LLM extraction test.")
        return
    
    try:
        # Create LlmConfig
        llm_config_obj = LlmConfig(
            provider=provider,
            api_token=api_token
        )
        
        # Create LLM extraction strategy
        llm_strategy = LLMExtractionStrategy(
            llm_config=llm_config_obj,
            instruction="Extract the main topics from this content.",
            output_format="json"
        )
        
        # Create crawler config
        config = CrawlerRunConfig(
            extraction_strategy=llm_strategy,
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
                
                # Try to display the extraction result
                if isinstance(result.extraction_result, str):
                    try:
                        parsed = json.loads(result.extraction_result)
                        print(f"Parsed JSON: {json.dumps(parsed, indent=2)[:500]}...")
                    except json.JSONDecodeError:
                        print(f"Raw extraction result (first 500 chars): {result.extraction_result[:500]}...")
                else:
                    print(f"Structured extraction result: {str(result.extraction_result)[:500]}...")
            else:
                print("No extraction result found.")
                
    except Exception as e:
        print(f"Error during LLM extraction test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the basic crawl test
    asyncio.run(test_basic_crawl())
    
    # Run the LLM extraction test if requested
    if "--with-llm" in sys.argv:
        asyncio.run(test_llm_extraction()) 