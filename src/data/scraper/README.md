# Political Figure Scraper

A powerful AI-powered web scraper for extracting structured data about political figures using BeautifulSoup and Ollama's Llama3 model.

## Overview

This scraper extracts comprehensive information about political figures from multiple sources including Wikipedia, Britannica, and Reuters. It leverages:

- Local LLM inference using Ollama/Llama3 (no API keys required)
- Efficient web scraping with BeautifulSoup
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

2. Start Ollama server (in a separate terminal):
```bash
ollama serve
```

3. Pull the Llama3 model:
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

You can specify a political figure as a command-line argument:

```bash
python politician_scraper.py "Joe Biden"
```

Or specify environment ID and GPU count:

```bash
python politician_scraper.py "Kamala Harris" nat 2
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
  ]
}
```

## Configuration Options

You can modify the following aspects of the scraper:

### Sources

Edit the `get_sources()` function to add or remove source URLs.

### Extraction Parameters

Adjust extraction parameters:

- `max_length` in `extract_with_ollama()` to control how much text is sent to Ollama
- Number of paragraphs to extract in `get_article_text()`

### Ollama Models

You can use different Ollama models by changing the model name in the requests:

```python
response = requests.post(
    'http://localhost:11434/api/generate',
    json={
        'model': 'mistral',  # Change model here
        'prompt': prompt,
        'stream': False
    }
)
```

## GPU Support

The script automatically attempts to set up a GPU environment using `genv`. To ensure proper GPU initialization:

```bash
eval "$(genv shell --init)"
```

Before running the script.

## Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   - Ensure Ollama server is running (`ollama serve`)
   - Verify the model is pulled (`ollama pull llama3`)

2. **No Data Extracted**
   - Check if the URLs are accessible
   - Try with a more well-known political figure

3. **GPU Environment Issues**
   - Initialize your shell with `eval "$(genv shell --init)"`
   - Ensure you have GPU access permissions

## License

MIT 