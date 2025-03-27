# -*- coding: utf-8 -*-

BOT_NAME = 'politician_crawler'

# Fix the module paths to ensure spiders can be discovered
SPIDER_MODULES = ['src.data.scraper.politician_crawler.spiders']
NEWSPIDER_MODULE = 'src.data.scraper.politician_crawler.spiders'

# Temporarily disable robots.txt rules for testing
ROBOTSTXT_OBEY = False

# Configure maximum concurrent requests
CONCURRENT_REQUESTS = 8
CONCURRENT_REQUESTS_PER_DOMAIN = 4

# Configure a delay for requests for the same website
DOWNLOAD_DELAY = 2.0
RANDOMIZE_DOWNLOAD_DELAY = True

# Disable cookies for tracking
COOKIES_ENABLED = False

# Configure item pipelines
ITEM_PIPELINES = {
   'src.data.scraper.politician_crawler.pipelines.SpacyNERPipeline': 300,
   'src.data.scraper.politician_crawler.pipelines.JsonWriterPipeline': 800,
}

# Enable and configure HTTP caching
HTTPCACHE_ENABLED = True
HTTPCACHE_EXPIRATION_SECS = 86400
HTTPCACHE_DIR = 'httpcache'
HTTPCACHE_IGNORE_HTTP_CODES = [503, 504, 403, 429, 500]

# Set log level to DEBUG for more verbose logging
LOG_LEVEL = 'DEBUG'

# User agent
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

# Retry settings
RETRY_ENABLED = True
RETRY_TIMES = 5
RETRY_HTTP_CODES = [500, 502, 503, 504, 408, 429]

# Timeout configuration
DOWNLOAD_TIMEOUT = 45  # Increased timeout to allow more time for responses
DOWNLOAD_TIMEOUTS = {
    'www.npr.org': 60,  # Increase timeout for NPR
    'www.cnn.com': 60,  # Increase for CNN
    'www.foxnews.com': 60  # Increase for Fox News
}

# Handle redirects better
REDIRECT_ENABLED = True
REDIRECT_MAX_TIMES = 5

# Failure handling
CLOSESPIDER_ERRORCOUNT = 50
CLOSESPIDER_TIMEOUT = 900  # 15 minutes max

# Custom settings for Politician Crawler
POLITICIAN_DATA_DIR = 'src/data/scraper/logs'
SPACY_MODEL = 'en_core_web_sm'  # Use small model by default to ensure availability

# Add more console logging to help debug
LOG_STDOUT = True

# Only try Wikipedia for initial testing
# This focuses the crawler on one reliable source for debugging
POLITICIAN_TEST_MODE = True 