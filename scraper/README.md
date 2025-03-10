# Politician Data Scraper

A Python script to scrape data about politicians using the crawl4ai tool, process the data, and store it in a Milvus vector database for AI political debate applications.

## Features

- Scrapes data about politicians from the web using crawl4ai
- Uses LLM-powered extraction to get structured data
- Cleans and preprocesses the data
- Stores the data in Milvus database for semantic search
- Supports custom search queries for different politicians
- Includes built-in configurations for Donald Trump and Joe Biden
- Easily extensible to other politicians

## Prerequisites

Before running the scraper, make sure you have:

1. Python 3.8+ installed
2. Milvus database running (see setup instructions in ../db/milvus/README.md)
3. Required Python packages installed (see Installation section)

## Installation

1. Install the required packages:

```bash
pip install -r scraper/requirements.txt
```

2. Set up Playwright for web browsing capability:

```bash
playwright install --with-deps chromium
```

3. Make sure Milvus is running. If not, start it using:

```bash
cd db/milvus
./setup.sh
```

## Usage

### Basic Usage

To scrape data about a politician and store it in the Milvus database:

```bash
python -m scraper.politician_scraper --name "Donald Trump"
```

or

```bash
python -m scraper.politician_scraper --name "Joe Biden"
```

### Other Politicians

The script can be used to scrape data about any politician:

```bash
python -m scraper.politician_scraper --name "Kamala Harris"
```

### Refreshing Data

To refresh existing data for a politician:

```bash
python -m scraper.politician_scraper --name "Donald Trump" --refresh
```

### Testing Without Database Interaction

To test the scraper without storing data in the database:

```bash
python -m scraper.test_scraper "Donald Trump"
```

### Batch Processing

To process multiple politicians at once:

```bash
python -m scraper.batch_scrape
```

or with custom list:

```bash
python -m scraper.batch_scrape --politicians "Donald Trump" "Joe Biden" "Kamala Harris"
```

## How It Works

1. **Search Queries**: The script generates search queries based on the politician's name or uses predefined queries for known politicians.

2. **Web Crawling**: It uses Google search to find relevant pages, then crawls these pages to extract content.

3. **Content Extraction**: 
   - Raw content is converted to Markdown format
   - An LLM strategy is used to extract structured data like biography, policy positions, etc.

4. **Data Processing**: The extracted data is processed and combined into a structured format.

5. **Database Storage**: The processed data is stored in Milvus vector database for semantic search.

## Customization

### Adding Custom Search Queries

To add custom search queries for specific politicians, modify the `POLITICIAN_SEARCH_QUERIES` dictionary in the script:

```python
POLITICIAN_SEARCH_QUERIES = {
    "Donald Trump": [
        "Donald Trump biography",
        # ... existing queries ...
    ],
    "Your Politician": [
        "Your Politician biography",
        "Your Politician political career",
        # ... more queries ...
    ]
}
```

### Modifying Extraction Prompts

You can modify the LLM extraction prompt in the `_extract_structured_data` method to extract different types of information.

## Project Structure

```
scraper/
├── __init__.py             # Package initialization
├── politician_scraper.py   # Main scraper class and script
├── test_scraper.py         # Test script without database interaction
├── batch_scrape.py         # Batch processing script
├── requirements.txt        # Package dependencies
└── README.md               # This documentation
```

## Troubleshooting

### Common Issues

1. **Milvus Connection Issues**:
   - Ensure Milvus is running (`docker ps`)
   - Check Milvus logs (`docker logs milvus`)

2. **Crawling Issues**:
   - Google might block requests if too many are made in quick succession
   - Try using a different user agent or adding delays between requests

3. **LLM Extraction Issues**:
   - The default local model might not be powerful enough for complex extraction
   - Consider using an OpenAI model by setting `use_local_model=False` and configuring your API key

### Logs

Check the logs for detailed error information:

```
logs/politician_scraper.log
logs/batch_scraper.log
```

## License

This project is licensed under the same license as the parent project.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 