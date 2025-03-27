#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Politician Data Scraper

A tool to collect and process information about political figures using:
- Scrapy for web crawling
- SpaCy for NER and text processing
- GPU acceleration for processing speed

Usage:
    python politician_scraper.py "Politician Name" [--output-dir DIR] [--env-id ENV_ID] [--gpu-count N]
"""

import os
import sys
import argparse
from politician_crawler.run_crawler import run_spider

def main():
    """Main function to parse arguments and run the scraper"""
    parser = argparse.ArgumentParser(
        description='Scrape political figure data from the web using Scrapy and SpaCy',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required argument
    parser.add_argument(
        'politician_name',
        type=str,
        help='Name of the political figure to scrape (e.g., "Donald Trump")'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/politicians',
        help='Directory to save the scraped data JSON files'
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
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the spider with provided arguments
    success = run_spider(
        politician_name=args.politician_name,
        output_dir=args.output_dir,
        env_id=args.env_id,
        gpu_count=args.gpu_count
    )
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    # Adjust path to allow importing from this package
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(current_dir))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    main()