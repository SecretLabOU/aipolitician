#!/usr/bin/env python3
"""
Run the Politician Spider to crawl data about a specified politician.

This script sets up the GPU environment (if available) and runs the Scrapy spider
to collect information about a political figure from various sources.

Usage:
    python run_crawler.py "Politician Name" [--output-dir DIR] [--env-id ENV_ID] [--gpu-count N]
"""

import os
import sys
import json
import time
import logging
import asyncio
import argparse
import subprocess
import tempfile
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crawler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add paths for imports - try both relative and absolute paths
try:
    # Try relative import first
    from .spiders.politician_spider import PoliticianSpider
except (ImportError, ValueError):
    try:
        # Try absolute import using src path
        from src.data.scraper.politician_crawler.spiders.politician_spider import PoliticianSpider
    except ImportError:
        # Last resort: modify sys.path and import
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        from spiders.politician_spider import PoliticianSpider

# Scrapy imports
try:
    import scrapy
    from scrapy.crawler import CrawlerProcess
    from scrapy.utils.project import get_project_settings
except ImportError:
    logger.error("Scrapy not installed. Please install with: pip install scrapy")
    raise

def setup_gpu_environment(env_id="nat", gpu_count=1) -> bool:
    """
    Set up GPU environment using genv.
    
    Args:
        env_id: The environment ID to use
        gpu_count: Number of GPUs to attach
        
    Returns:
        bool: True if setup was successful
    """
    if gpu_count <= 0:
        logger.info("GPU acceleration disabled")
        return True
        
    try:
        logger.info(f"Setting up GPU environment (env_id={env_id}, gpus={gpu_count})...")
        
        # Try to set up the genv environment
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            script_path = f.name
            f.write("""#!/bin/bash
# Initialize genv
source ~/.bashrc
if command -v genv &> /dev/null; then
    echo "genv is available"
else
    echo "genv command not found"
    exit 1
fi

# Create and activate environment
genv init ${ENV_ID} || true
genv shell ${ENV_ID}

# Attach GPU
genv gpu attach ${DEVICE_ID}

# Check GPU status
echo "GPU Status:"
nvidia-smi
""")
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        # Run the script with environment variables
        result = subprocess.run(
            ["bash", script_path],
            env={
                **os.environ,
                "ENV_ID": env_id,
                "DEVICE_ID": str(gpu_count)
            },
            capture_output=True,
            text=True
        )
        
        # Clean up the temporary script
        os.unlink(script_path)
        
        # Check if successful
        if result.returncode != 0:
            logger.error(f"GPU setup failed with exit code {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            return False
            
        logger.info(result.stdout)
        logger.info("GPU environment set up successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error setting up GPU environment: {e}")
        return False
        
def cleanup_gpu_environment(env_id="nat") -> bool:
    """
    Clean up GPU environment by detaching GPUs.
    
    Args:
        env_id: The environment ID to clean up
        
    Returns:
        bool: True if cleanup was successful
    """
    try:
        logger.info(f"Cleaning up GPU environment (env_id={env_id})...")
        
        # Create a temporary shell script to clean up the GPU environment
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            script_path = f.name
            f.write("""#!/bin/bash
# Initialize genv
source ~/.bashrc
if command -v genv &> /dev/null; then
    echo "genv is available"
else
    echo "genv command not found"
    exit 1
fi

# Activate environment
genv shell ${ENV_ID}

# Detach GPUs
genv gpu detach all

echo "GPU environment cleaned up"
""")
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        # Run the script with environment variables
        result = subprocess.run(
            ["bash", script_path],
            env={
                **os.environ,
                "ENV_ID": env_id
            },
            capture_output=True,
            text=True
        )
        
        # Clean up the temporary script
        os.unlink(script_path)
        
        return True
    except Exception as e:
        logger.error(f"Error cleaning up GPU environment: {e}")
        return False

async def run_spider(politician_name, output_dir=None, env_id="nat", gpu_count=1) -> Optional[Dict[str, Any]]:
    """
    Run the Scrapy spider to collect data about a political figure.
    
    Args:
        politician_name: Name of the politician to scrape
        output_dir: Directory to save the scraped data
        env_id: GPU environment ID for genv
        gpu_count: Number of GPUs to use (0 for CPU only)
        
    Returns:
        Dict containing scraped data or None if unsuccessful
    """
    try:
        # Start timing
        start_time = time.time()
        
        # Set default output directory if not specified
        if not output_dir:
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup GPU environment if requested
        if gpu_count > 0:
            gpu_success = setup_gpu_environment(env_id, gpu_count)
            if not gpu_success:
                logger.warning("GPU setup failed, continuing without GPU acceleration")
        
        # Generate output filename - sanitize politician name to avoid path issues
        safe_name = politician_name.lower().replace(' ', '_')
        # Remove any other unsafe characters
        safe_name = ''.join(c for c in safe_name if c.isalnum() or c == '_')
        filename = f"{safe_name}.json"
        output_file = os.path.join(output_dir, filename)
        
        # Initialize crawler settings
        settings = get_project_settings()
        settings.update({
            'FEEDS': {
                output_file: {
                    'format': 'json',
                    'encoding': 'utf8',
                    'overwrite': True,
                },
            },
            'LOG_LEVEL': 'INFO',
            'LOG_FILE': f"{output_dir}/spider.log",
            'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'ROBOTSTXT_OBEY': False,
            'DOWNLOAD_DELAY': 1.0,
            'CONCURRENT_REQUESTS': 16,
            'COOKIES_ENABLED': False,
            'RETRY_TIMES': 3,
            'RETRY_HTTP_CODES': [500, 502, 503, 504, 408, 429],
            'HTTPERROR_ALLOW_ALL': True
        })
        
        logger.info(f"Starting crawler for: {politician_name}")
        logger.info(f"Output file: {output_file}")
        
        # Use CrawlerRunner instead of CrawlerProcess for better resource management
        # and isolation between different crawls
        from scrapy.crawler import CrawlerRunner
        from twisted.internet import reactor, defer
        from twisted.internet.error import ReactorNotRestartable
        
        # Use the synchronous run_crawler helper to run the async Twisted code
        data = await run_crawler_with_reactor(politician_name, settings, output_file)
        
        # Calculate duration
        duration = time.time() - start_time
        logger.info(f"Crawler finished in {duration:.2f} seconds")
        
        # Clean up GPU environment if it was used
        if gpu_count > 0:
            cleanup_gpu_environment(env_id)
        
        return data
            
    except Exception as e:
        logger.error(f"Error running spider: {e}")
        if gpu_count > 0:
            cleanup_gpu_environment(env_id)
        return None

async def run_crawler_with_reactor(politician_name, settings, output_file):
    """
    Run a crawler with its own reactor to ensure isolation between crawls.
    
    Args:
        politician_name: Name of the politician to scrape
        settings: Scrapy settings dictionary
        output_file: Path to the output file
        
    Returns:
        Scraped data or None if unsuccessful
    """
    try:
        from scrapy.crawler import CrawlerRunner
        from twisted.internet import reactor, defer
        
        # Create a deferred to signal when crawl is complete
        crawl_complete = defer.Deferred()
        
        # Create a new crawler runner with the specified settings
        runner = CrawlerRunner(settings)
        
        @defer.inlineCallbacks
        def crawl():
            try:
                yield runner.crawl(PoliticianSpider, politician_name=politician_name)
                crawl_complete.callback(None)
            except Exception as e:
                logger.error(f"Error in crawler: {e}")
                crawl_complete.errback(e)
        
        # Start the crawl
        crawl()
        
        # Wait for the crawl to complete using an asyncio Future
        future = asyncio.Future()
        
        def on_crawl_success(_):
            if not future.done():
                future.set_result(True)
        
        def on_crawl_failure(failure):
            if not future.done():
                future.set_exception(Exception(f"Crawl failed: {failure}"))
        
        crawl_complete.addCallbacks(on_crawl_success, on_crawl_failure)
        
        # Run the reactor in a separate thread until the crawl completes
        def run_reactor():
            if not reactor.running:
                reactor.run(installSignalHandlers=False)
                
        import threading
        reactor_thread = threading.Thread(target=run_reactor)
        reactor_thread.daemon = True
        reactor_thread.start()
        
        try:
            # Wait for the crawl to complete
            await future
            
            # Stop the reactor
            if reactor.running:
                reactor.callFromThread(reactor.stop)
                
            # Wait for the thread to finish
            reactor_thread.join(timeout=5.0)
            
            # Load the scraped data from the output file
            if os.path.exists(output_file):
                try:
                    with open(output_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    # Check if the data is a list (Scrapy's default output format)
                    if isinstance(data, list) and len(data) > 0:
                        return data  # Return all items, not just the first
                    elif isinstance(data, dict):
                        return data
                    else:
                        logger.error("Invalid data format in crawler output")
                        return None
                except json.JSONDecodeError:
                    logger.error(f"Error parsing JSON from {output_file}")
                    return None
                except Exception as e:
                    logger.error(f"Error loading scraped data: {e}")
                    return None
            else:
                logger.error(f"Output file not found: {output_file}")
                return None
                
        except Exception as e:
            logger.error(f"Error waiting for crawl to complete: {e}")
            # Ensure reactor stops
            if reactor.running:
                reactor.callFromThread(reactor.stop)
            return None
        
    except Exception as e:
        logger.error(f"Error setting up crawler: {e}")
        return None

def main():
    """Main function to parse arguments and run the spider"""
    parser = argparse.ArgumentParser(
        description='Run the Politician Spider to collect data about a political figure',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        'politician_name',
        type=str,
        help='Name of the political figure to scrape'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save the scraped data'
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
    
    args = parser.parse_args()
    
    # Run the spider
    asyncio.run(run_spider(
        politician_name=args.politician_name,
        output_dir=args.output_dir,
        env_id=args.env_id,
        gpu_count=args.gpu_count
    ))

if __name__ == "__main__":
    main() 