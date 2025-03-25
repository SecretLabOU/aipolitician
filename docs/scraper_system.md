# Scraper System Documentation

The AI Politician system includes a web scraper component that collects politician data from reliable sources. This document explains how the scraper works and how to use it.

## Overview

The scraper is designed to collect data such as:
- Speeches
- Statements
- Press releases
- Interviews
- Policy positions
- Biographical information

from various sources including government websites, official campaign sites, news outlets, and more.

## Scraper Components

The scraper is implemented in `src/data/scraper/politician_scraper.py` and contains:

1. Source-specific scraper classes
2. Content extraction utilities
3. Rate limiting and politeness controls
4. Error handling and recovery mechanisms

## Supported Sources

The scraper can collect data from multiple sources, including:

- Official government websites (e.g., whitehouse.gov)
- Campaign websites
- Social media platforms
- News outlets
- Public speech repositories
- Official document archives

## Usage

### Basic Usage

To run the scraper for a specific politician:

```bash
python -m src.data.scraper.politician_scraper --politician biden --output-dir ./data/raw/biden
```

### Configuration Options

The scraper supports several configuration options:

- `--politician`: Target politician (biden, trump)
- `--output-dir`: Directory to store scraped data
- `--start-date`: Date to start scraping from (YYYY-MM-DD)
- `--end-date`: Date to end scraping at (YYYY-MM-DD)
- `--sources`: Comma-separated list of sources to scrape
- `--max-items`: Maximum number of items to scrape per source
- `--cache`: Use cached data when available
- `--verbose`: Enable verbose logging

### Example Commands

```bash
# Scrape Biden data with date range
python -m src.data.scraper.politician_scraper --politician biden --output-dir ./data/raw/biden --start-date 2021-01-20 --end-date 2023-01-20

# Scrape Trump data from specific sources
python -m src.data.scraper.politician_scraper --politician trump --output-dir ./data/raw/trump --sources whitehouse,campaign,twitter
```

## Output Format

The scraper outputs data in JSON format, with each file containing:

```json
{
  "content": "Text content of the speech/statement",
  "title": "Title of the document",
  "date": "Publication date",
  "source": "Original source URL",
  "type": "Document type (speech, statement, etc.)",
  "metadata": {
    "location": "Location if applicable",
    "audience": "Target audience if known",
    "topics": ["List", "of", "topics"],
    "additional_fields": "..."
  }
}
```

## Adding New Sources

To add a new source to the scraper:

1. Create a new scraper class in `politician_scraper.py` that inherits from the base scraper
2. Implement the required methods: `fetch`, `parse`, and `extract`
3. Add the new source to the source registry
4. Update the CLI argument parser to include the new source

## Requirements

The scraper requires several libraries listed in `requirements/requirements-scraper.txt`, including:

- requests
- beautifulsoup4
- selenium (for JavaScript-heavy sites)
- aiohttp (for async scraping)
- playwright (for complex sites)

Install these dependencies using:

```bash
pip install -r requirements/requirements-scraper.txt
```

## Logging

The scraper logs its activity to `src/data/scraper/logs/` with different log levels:

- INFO: Basic scraping progress
- DEBUG: Detailed scraping information
- WARNING: Issues that don't prevent operation
- ERROR: Problems that prevent successful scraping

You can view the logs to monitor progress and troubleshoot issues. 