#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import subprocess
import atexit
import time
import datetime
import logging
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('scraper.log')
    ]
)
logger = logging.getLogger(__name__)

def setup_gpu_environment(env_id="nat", gpu_count=1):
    """Setup genv environment and attach GPU"""
    logger.info(f"Setting up GPU environment (id: {env_id}, GPUs: {gpu_count})...")
    
    try:
        # First check if genv is properly initialized
        init_cmd = "eval \"$(genv shell --init)\" && echo 'GENV_INITIALIZED'"
        result = subprocess.run(init_cmd, shell=True, check=False, capture_output=True, text=True)
        
        if "GENV_INITIALIZED" not in result.stdout:
            logger.warning("⚠️ genv shell initialization required. Please run: eval \"$(genv shell --init)\"")
            return False
            
        # Check if environment is already active
        status_cmd = "genv status"
        status_result = subprocess.run(status_cmd, shell=True, check=False, capture_output=True, text=True)
        
        # If environment is already active with the same ID, don't try to activate it again
        if f"Environment ID: {env_id}" in status_result.stdout:
            logger.info(f"✅ Environment '{env_id}' is already active")
        else:
            # Activate the environment with proper shell initialization
            activate_cmd = f"eval \"$(genv shell --init)\" && genv activate --id {env_id}"
            activate_result = subprocess.run(activate_cmd, shell=True, check=True, capture_output=True, text=True)
            logger.info(f"✅ Activated genv environment: {env_id}")
        
        # Check if GPUs are already attached
        devices_cmd = "genv devices"
        devices_result = subprocess.run(devices_cmd, shell=True, check=False, capture_output=True, text=True)
        
        # Only attach GPUs if needed
        if "No devices attached" in devices_result.stdout:
            # Attach GPUs
            attach_cmd = f"genv attach --count {gpu_count}"
            subprocess.run(attach_cmd, shell=True, check=True)
            logger.info(f"✅ Attached {gpu_count} GPU(s) to environment")
        else:
            logger.info("✅ GPUs are already attached to the environment")
        
        # Show GPU status
        subprocess.run("nvidia-smi", shell=True, check=False)
        
        # Register cleanup on exit
        atexit.register(cleanup_gpu_environment)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to setup GPU environment: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error setting up GPU: {e}")
        return False

def cleanup_gpu_environment():
    """Clean up by detaching GPUs"""
    try:
        logger.info("Cleaning up GPU environment...")
        # Detach all GPUs with proper shell initialization
        subprocess.run("eval \"$(genv shell --init)\" && genv detach --all", shell=True, check=True)
        logger.info("✅ Successfully detached all GPUs")
    except Exception as e:
        logger.warning(f"⚠️ Error during GPU cleanup: {e}")

def run_spider(politician_name, output_dir=None, env_id="nat", gpu_count=1):
    """Run the Scrapy spider for the given politician name with GPU support"""
    start_time = time.time()
    logger.info(f"Starting crawler for politician: {politician_name}")
    
    # Setup GPU environment if requested
    if gpu_count > 0:
        if not setup_gpu_environment(env_id, gpu_count):
            logger.warning("⚠️ GPU setup failed, continuing with CPU only")
    else:
        logger.info("Running with CPU only (no GPU requested)")
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        # Update settings to use this directory
        os.environ['POLITICIAN_DATA_DIR'] = output_dir
    
    try:
        # Get the Scrapy project settings
        settings = get_project_settings()
        
        # Override settings if necessary
        if output_dir:
            settings.set('POLITICIAN_DATA_DIR', output_dir)
        
        # Set up the crawler process
        process = CrawlerProcess(settings)
        
        # Add the spider
        process.crawl('politician', politician_name=politician_name)
        
        # Start crawling
        process.start()  # This blocks until crawling is finished
        
        # Calculate and report runtime
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Crawling finished in {duration:.2f} seconds")
        
        return True
    except Exception as e:
        logger.error(f"Error running spider: {e}")
        return False
    finally:
        # No need to call cleanup_gpu_environment here as it's registered with atexit
        pass

def main():
    """Parse command line arguments and run the crawler"""
    parser = argparse.ArgumentParser(description='Crawl web data for a political figure using Scrapy and SpaCy')
    parser.add_argument('politician_name', type=str, help='Name of the politician to crawl data for')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save the output files')
    parser.add_argument('--env-id', type=str, default='nat', help='GPU environment ID for genv')
    parser.add_argument('--gpu-count', type=int, default=1, help='Number of GPUs to use (0 for CPU only)')
    
    args = parser.parse_args()
    
    # Run the crawler
    success = run_spider(
        politician_name=args.politician_name,
        output_dir=args.output_dir,
        env_id=args.env_id,
        gpu_count=args.gpu_count
    )
    
    if success:
        logger.info("Crawling completed successfully")
        sys.exit(0)
    else:
        logger.error("Crawling failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 