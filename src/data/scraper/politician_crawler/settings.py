# -*- coding: utf-8 -*-

BOT_NAME = 'politician_crawler'

# Fix the module paths to ensure spiders can be discovered
SPIDER_MODULES = ['src.data.scraper.politician_crawler.spiders']
NEWSPIDER_MODULE = 'src.data.scraper.politician_crawler.spiders'

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

# Configure maximum concurrent requests
CONCURRENT_REQUESTS = 16
CONCURRENT_REQUESTS_PER_DOMAIN = 8

# Configure a delay for requests for the same website
DOWNLOAD_DELAY = 1.5
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

# Set log level
LOG_LEVEL = 'INFO'

# User agent
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

# Retry settings
RETRY_ENABLED = True
RETRY_TIMES = 3
RETRY_HTTP_CODES = [500, 502, 503, 504, 408, 429]

# Custom settings for Politician Crawler
POLITICIAN_DATA_DIR = 'src/data/scraper/logs'
SPACY_MODEL = 'en_core_web_lg'  # Transformer-based model for better NER 