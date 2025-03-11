# Political Figure Scraper

A powerful AI-powered web scraper for extracting structured data about political figures using the `crawl4ai` package and Ollama's Llama3 model.

## Overview

This scraper extracts comprehensive information about political figures from multiple sources including Wikipedia, Britannica, and Reuters. It leverages:

- Local LLM inference using Ollama/Llama3 (no API keys required)
- Intelligent content extraction and structuring
- Multi-source aggregation for comprehensive data collection

## Installation

### Prerequisites

1. Python 3.8+ installed
2. [Ollama](https://ollama.com/) installed on your system

### Setup

1. Install Python dependencies:
```bash
cd scraper
pip install -r requirements.txt
```

2. Install required browser binaries:
```bash
python -m playwright install
```

3. Start Ollama server (in a separate terminal):
```bash
ollama serve
```

4. Pull the Llama3 model:
```bash
ollama pull llama3
```

## Usage

### Basic Usage

Run the main scraper:

```bash
python politician_scraper.py
```

This will scrape data for Barack Obama (default example) and save it to `scraper/logs/`.

### Custom Political Figure

Modify the `politician_scraper.py` file to change the target political figure:

```python
if __name__ == "__main__":
    political_figure_name = "Joe Biden"  # Change to any political figure
    main(political_figure_name)
```

### Quick Testing

For a faster test that only scrapes Wikipedia:

```bash
python test_scraper.py
```

## Output Data

The scraper produces a JSON file with the following structure:

```json
{
  "id": "unique-uuid",
  "name": "Political Figure Name",
  "date_of_birth": "Month Day, Year",
  "nationality": "Country",
  "political_affiliation": "Party",
  "biography": "Comprehensive biography text...",
  "positions": [
    "List of political positions held"
  ],
  "policies": [
    "List of policy positions"
  ],
  "legislative_actions": [
    "List of notable legislative actions"
  ],
  "campaigns": [
    "List of notable campaigns"
  ],
  "achievements": [
    "List of achievements"
  ]
}
```

## Configuration Options

You can modify the following aspects of the scraper:

### Sources

Edit the `get_sources()` function to add or remove source URLs.

### Extraction Parameters

Adjust extraction parameters in `crawl_political_figure()`:

- `max_depth`: How many links deep to crawl (default: 1)
- `max_pages`: Maximum pages to crawl per source (default: 3)
- `chunk_token_threshold`: Token limit for LLM processing (default: 4000)

### LLM Configuration

You can use different Ollama models by changing:

```python
llm_config = LLMConfig(
    provider="ollama/mistral",  # Change model here
    base_url="http://localhost:11434"
)
```

## Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   - Ensure Ollama server is running (`ollama serve`)
   - Verify the model is pulled (`ollama pull llama3`)

2. **No Data Extracted**
   - Check if the URLs are accessible
   - Try with a more well-known political figure
   - Adjust the `word_count_threshold` to a lower value

3. **Slow Performance**
   - Local LLM inference can be slow depending on your hardware
   - Reduce `max_pages` and `max_depth` for faster results
   - Use the test_scraper.py for quicker testing

## License

MIT 