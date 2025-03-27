# Political Figure Scraper Documentation

## Overview

The Political Figure Scraper is a sophisticated tool designed to collect comprehensive factual information about political figures from diverse authoritative sources. It leverages both web scraping technologies and Local Language Models to extract, structure, and enhance political data.

---

## 🔍 Technical Architecture

### Key Technologies

- **Python** with asynchronous programming via `asyncio`
- **BeautifulSoup** for HTML parsing and content extraction
- **SentenceTransformer** for generating text embeddings
- **Ollama** with **Llama3** for structured information extraction
- **GPU acceleration** with custom environment management

### Directory Structure

```
src/data/scraper/
├── __init__.py              # Package initialization
├── politician_scraper.py    # Main scraper implementation
├── README.md                # Basic usage documentation
├── logs/                    # Output and log directory
└── __pycache__/             # Python cache files
```

### Scraper Workflow

The scraper follows a systematic process:

1. **Source Collection**: Dynamically generates appropriate URLs for a given politician
2. **Content Fetching**: Retrieves raw HTML with retry logic and error handling
3. **Content Extraction**: Parses HTML to extract relevant text content
4. **Structured Information Extraction**: Uses Ollama to transform raw text into structured JSON
5. **Embedding Generation**: Creates vector embeddings for the extracted content
6. **Output Generation**: Formats and saves the results to the logs directory

---

## 💻 Implementation Details

### Source Generation

The scraper dynamically generates URLs for multiple source types:

```python3
def get_sources(name):
    formatted_name = name.replace(' ', '_')
    formatted_name_dash = name.replace(' ', '-')
    formatted_name_plus = name.replace(' ', '+')
    
    return [
        # Wikipedia and encyclopedias
        f"https://en.wikipedia.org/wiki/{formatted_name}",
        f"https://www.britannica.com/biography/{formatted_name_dash}",
        
        # News sources
        f"https://www.reuters.com/search/news?blob={name}",
        f"https://apnews.com/search?q={formatted_name_plus}",
        # ... additional sources
    ]
```

### Content Extraction

Web content is extracted using BeautifulSoup with site-specific selectors and robust error handling:

```python3
def get_article_text(url, selector=None, max_retries=2):
    """Fetch and extract the main content from a webpage with retries"""
    # Implementation with retry logic and error handling
```

### Structured Information Extraction

The scraper leverages Ollama's Llama3 model to extract structured information from raw text:

```python3
def extract_with_ollama(text, name, max_length=8000):
    """Extract structured information directly using Ollama API"""
    # Trims text to reasonable length
    # Prepares a comprehensive prompt for Llama3
    # Handles API communication and error cases
```

The extraction prompt defines a detailed schema for organizing political information:

```json
{
    "basic_info": {
        "full_name": "Complete name including middle names",
        "date_of_birth": "YYYY-MM-DD format",
        "place_of_birth": "City, State/Province, Country",
        "nationality": "Country of citizenship",
        "political_affiliation": "Political party or affiliation",
        "education": ["Educational qualifications with institutions and years"],
        "family": ["Purely factual family information"]
    },
    "career": {
        "positions": ["All political positions held with dates"],
        "pre_political_career": ["Previous occupations before entering politics"],
        "committees": ["Committee memberships"]
    },
    // Additional categories omitted for brevity
}
```

### GPU Acceleration

The scraper can utilize GPU acceleration for improved performance:

```python3
def setup_gpu_environment(env_id="nat", gpu_count=1):
    """Setup genv environment and attach GPU"""
    # Implementation for GPU environment setup
```

---

## 🚀 Usage Guide

### Prerequisites

1. Python 3.8+ installed
2. [Ollama](https://ollama.com/) installed and running
3. Required Python packages:
   - beautifulsoup4
   - requests
   - sentence-transformers
   - numpy

### Installation

#### Setting Up a Conda Environment

1. Create a new conda environment named "scraper":
   ```bash
   conda create -n scraper python=3.8
   ```

2. Activate the environment:
   ```bash
   conda activate scraper
   ```

3. Navigate to the project root directory:
   ```bash
   cd /path/to/ai-politician
   ```

4. Install required dependencies:
   ```bash
   pip install -r requirements/requirements-scraper.txt
   ```

   The scraper has the following core dependencies:
   - requests: For HTTP requests
   - beautifulsoup4: For HTML parsing
   - numpy: For numerical operations
   - sentence-transformers: For generating text embeddings

5. Install Ollama:
   - For macOS: Download from [ollama.com/download](https://ollama.com/download)
   - For Linux:
     ```bash
     curl -fsSL https://ollama.com/install.sh | sh
     ```

6. Pull the Llama3 model:
   ```bash
   ollama pull llama3
   ```

7. Start the Ollama server (in a separate terminal):
   ```bash
   ollama serve
   ```

### Basic Usage

Run the scraper with default settings:

```bash
# Make sure your conda environment is activated
conda activate scraper

# Start the Ollama server (in a separate terminal)
ollama serve

# Run the scraper (uses default politician "Donald Trump")
python3 -m src.data.scraper.politician_scraper
```

### Custom Configuration

The script uses positional arguments rather than flags:

```bash
python3 -m src.data.scraper.politician_scraper "Politician Name" "env_id" gpu_count
```

For example, to scrape data for JD Vance with environment ID "nat" and 1 GPU:

```bash
python3 -m src.data.scraper.politician_scraper "JD Vance" "nat" 1
```

### Command-line Arguments

| Position | Parameter | Description | Default |
|----------|-----------|-------------|---------|
| 1 | Politician name | Name of the political figure to scrape | "Donald Trump" |
| 2 | Environment ID | GPU environment ID | "nat" |
| 3 | GPU count | Number of GPUs to use | 1 |

### GPU Environment Setup

Before running the script with GPU support:

1. Initialize genv in your shell:
   ```bash
   eval "$(genv shell --init)"
   ```

2. Create a genv environment if you don't have one already:
   ```bash
   genv create --id nat
   ```

The script will attempt to:
- Activate the specified genv environment
- Attach the requested number of GPUs
- Clean up by detaching GPUs when the script exits

---

## 📊 Output Format

The scraper produces structured JSON output:

```json
{
  "id": "unique-uuid",
  "name": "Political Figure Name",
  "basic_info": {
    "full_name": "Complete formal name",
    "date_of_birth": "YYYY-MM-DD",
    "place_of_birth": "Location",
    "nationality": "Country",
    "political_affiliation": "Party",
    "education": ["Education details"],
    "family": ["Family information"]
  },
  "career": {
    "positions": ["Political positions with dates"],
    "pre_political_career": ["Prior career details"],
    "committees": ["Committee memberships"]
  },
  "policy_positions": {
    "economy": ["Economic policy positions"],
    "foreign_policy": ["Foreign policy positions"],
    "healthcare": ["Healthcare policy positions"],
    // Additional policy areas
  },
  // Additional categories (legislative_record, communications, etc.)
  "embedding": [0.123, -0.456, ...] // Vector representation
}
```

---

## 🛠️ Customization

### Adding New Sources

Edit the `get_sources()` function to include additional sources:

```python3
def get_sources(name):
    # Existing sources
    sources = [...]
    
    # Add new sources
    sources.append(f"https://new-source.com/search?q={name}")
    
    return sources
```

### Modifying Extraction Schema

Edit the prompt in `extract_with_ollama()` to modify the extraction schema:

```python3
prompt = f"""
Extract ONLY factual information about {name} from this text.

Return a JSON object with THESE EXACT fields:
{{
    "basic_info": {{
        // Modify or add fields here
    }},
    // Add or modify categories
}}
"""
```

### Using Different LLM Models

Change the model name in the Ollama API call:

```python3
response = requests.post(
    'http://localhost:11434/api/generate',
    json={
        'model': 'mistral', # Change from llama3 to another model
        'prompt': prompt,
        'stream': False
    }
)
```

---

## 📝 Troubleshooting

### Common Issues

1. **Ollama Connection Issues**
   - Ensure Ollama is running with `ollama serve`
   - Verify the model is available with `ollama list`

2. **Content Extraction Failures**
   - Check if website structure has changed
   - Review logs for specific error messages
   - Try different selectors or extraction methods

3. **GPU Environment Issues**
   - Ensure GPU drivers are properly installed
   - Verify `genv` is correctly configured
   - Check GPU availability with `nvidia-smi`

### Logging

The scraper outputs log files to the `logs/` directory with:
- Execution timestamps
- URLs processed
- Content extraction results
- Error messages and warnings
- Performance metrics

---

## 🔄 Maintenance and Updates

### Adding Support for New Political Figures

The scraper is designed to work with any political figure by simply providing their name. However, for optimal results:

1. Check if the political figure has consistent naming across sources
2. Verify that sufficient information is available online
3. Consider adding custom selectors for politician-specific sources

### Updating Source Selectors

Web page structures may change over time. To update selectors:

1. Inspect the relevant website using browser developer tools
2. Identify new CSS selectors or XPath expressions for content
3. Update the site-specific extraction logic in `get_article_text()`