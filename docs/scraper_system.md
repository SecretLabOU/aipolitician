# Political Figure Scraper Documentation

## Overview

The Political Figure Scraper is a sophisticated tool designed to collect comprehensive factual information about political figures from diverse authoritative sources. It leverages Scrapy for web crawling and SpaCy for natural language processing to extract, structure, and enhance political data.

---

## 🔍 Technical Architecture

### Key Technologies

- **Scrapy** for robust, concurrent web crawling
- **SpaCy** with GPU acceleration for Named Entity Recognition (NER) and text processing
- **python3** with modern project structure
- **GPU acceleration** with genv environment management

### Directory Structure

```
src/data/scraper/
├── __init__.py                       # Package initialization
├── politician_scraper.py             # Main entry script
├── politician_crawler/               # Scrapy project
│   ├── __init__.py                   # Package initialization
│   ├── items.py                      # Data structure definitions
│   ├── pipelines.py                  # Processing pipelines (SpaCy NER)
│   ├── settings.py                   # Scrapy settings
│   ├── run_crawler.py                # Crawler execution script
│   └── spiders/                      # Scrapy spiders
│       ├── __init__.py               # Package initialization
│       └── politician_spider.py      # Main politician spider
├── README.md                         # Basic usage documentation
└── logs/                             # Log files directory
```

### Scraper Workflow

The scraper follows a systematic process:

1. **Web Crawling**: Scrapy spider navigates through multiple sources for a given politician
2. **Content Extraction**: Domain-specific parsers extract relevant content from each source
3. **NER Processing**: SpaCy performs named entity recognition to extract structured data
4. **Data Integration**: Information from different sources is consolidated
5. **Output Generation**: Final results are stored in JSON format

---

## 💻 Implementation Details

### Scrapy Spider

The system uses a custom Scrapy spider designed to handle various political information sources:

```python3
class PoliticianSpider(scrapy.Spider):
    name = "politician"
    allowed_domains = [
        'en.wikipedia.org',
        'www.britannica.com',
        'www.reuters.com',
        # Additional domains...
    ]
    
    def __init__(self, politician_name=None, *args, **kwargs):
        # Initialize with politician name
        # Generate appropriate URLs based on formatted names
```

The spider includes specialized parsers for different website types:

- Wikipedia articles
- News sources
- Government databases
- Speech archives
- Fact-checking sites

### SpaCy NER Processing

A custom pipeline uses SpaCy's powerful NER capabilities:

```python3
class SpacyNERPipeline:
    def process_item(self, item, spider):
        # Process text with SpaCy
        doc = self.nlp(item['raw_content'])
        
        # Extract named entities by type
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                person_entities.append(ent.text)
            elif ent.label_ == "ORG":
                org_entities.append(ent.text)
            # Additional entity types...
```

The pipeline automatically:
- Detects GPU availability and optimizes processing
- Extracts person names, organizations, dates, locations, and events
- Identifies potential birth dates and political affiliations
- Deduplicates extracted information

### GPU Acceleration

The scraper dynamically adapts to available GPU resources:

```python3
# Check for GPU availability and optimize settings
has_gpu = len(get_cuda_devices()) > 0
if has_gpu:
    spacy.prefer_gpu()
    # GPU-optimized settings
```

The GPU environment is managed through genv:
- Automatic initialization and cleanup
- Support for multiple GPUs
- Fallback to CPU processing when GPUs are unavailable

---

## 🚀 Usage Guide

### Prerequisites

1. python3 3.8+ installed
2. Required python3 packages:
   - scrapy
   - spacy
   - requests
   - beautifulsoup4

### Installation

#### Setting Up the Environment

1. Create a new conda environment named "scraper":
   ```bash
   conda create -n scraper python3=3.8
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

5. Download the SpaCy model:
   ```bash
   python3 -m spacy download en_core_web_trf
   ```
   
   For lower resource usage, you can use alternative models:
   ```bash
   # Large model - good balance between accuracy and performance
   python3 -m spacy download en_core_web_lg
   
   # Small model - fastest but less accurate
   python3 -m spacy download en_core_web_sm
   ```

### GPU Environment Setup (Optional)

Before running the script with GPU support:

1. Initialize genv in your shell:
   ```bash
   eval "$(genv shell --init)"
   ```

2. Activate a genv environment:
   ```bash
   genv activate --id nat
   ```

The script will:
- Check if genv is properly initialized
- Verify if the environment is already active
- Check if GPUs are already attached
- Clean up by detaching GPUs when the script exits

### Basic Usage

Run the scraper with default settings:

```bash
# Make sure your conda environment is activated
conda activate scraper

# Run the scraper with default settings
python3 -m src.data.scraper.politician_scraper "Donald Trump"
```

### Command-line Arguments

```bash
python3 -m src.data.scraper.politician_scraper "Politician Name" [--output-dir DIR] [--env-id ENV_ID] [--gpu-count N]
```

| Argument | Description | Default |
|----------|-------------|---------|
| politician_name | Name of the political figure to scrape | (Required) |
| --output-dir | Directory to save the output files | data/politicians |
| --env-id | GPU environment ID | nat |
| --gpu-count | Number of GPUs to use (0 for CPU only) | 1 |

For example, to scrape data for Joe Biden with environment ID "nat" and 2 GPUs:

```bash
python3 -m src.data.scraper.politician_scraper "Joe Biden" --output-dir data/politicians --env-id nat --gpu-count 2
```

---

## 📊 Output Format

The scraper produces structured JSON output with entities extracted by SpaCy:

```json
{
  "id": "unique-uuid",
  "name": "Political Figure Name",
  "source_url": "Source URL",
  "full_name": "Complete formal name",
  "date_of_birth": "Extracted birth date",
  "political_affiliation": "Detected political party",
  "raw_content": "Original text content",
  "person_entities": ["List of detected person names"],
  "org_entities": ["List of detected organizations"],
  "date_entities": ["List of detected dates"],
  "gpe_entities": ["List of detected geopolitical entities"],
  "event_entities": ["List of detected events"],
  "sponsored_bills": ["Bills sponsored by the politician"],
  "voting_record": ["Voting history entries"],
  "speeches": ["Speech transcripts"],
  "statements": ["Public statements"],
  "timestamp": "ISO timestamp of extraction"
}
```

---

## 🛠️ Customization

### Adding New Source Domains

Edit the `allowed_domains` and `generate_start_urls()` method in `politician_spider.py`:

```python3
allowed_domains = [
    'en.wikipedia.org',
    'www.britannica.com',
    # Add your new domain here
    'new-domain.com'
]

def generate_start_urls(self):
    # Add your new URL pattern
    return [
        # Existing URLs
        f"https://new-domain.com/profile/{self.formatted_name_dash}"
    ]
```

### Creating Custom Parsers

Add a new parser method in `politician_spider.py`:

```python3
def parse_custom_site(self, response):
    """Parse your custom site"""
    main_content = response.css('your-css-selector')
    
    if not main_content:
        self.logger.warning(f"No content found on custom page: {response.url}")
        return
    
    # Extract content
    paragraphs = main_content.css('p, li').getall()
    raw_content = "\n".join([self.clean_html(p) for p in paragraphs])
    
    # Create item
    item = PoliticianItem()
    item['name'] = self.politician_name
    item['source_url'] = response.url
    item['raw_content'] = raw_content
    
    # Custom field extraction
    special_info = response.css('special-selector::text').get()
    if special_info:
        item['your_custom_field'] = special_info
    
    yield item
```

### Using Different SpaCy Models

Edit the `SPACY_MODEL` setting in `settings.py`:

```python3
# For highest accuracy (requires more GPU memory)
SPACY_MODEL = 'en_core_web_trf'

# For good balance between accuracy and performance
# SPACY_MODEL = 'en_core_web_lg'

# For fastest performance but lower accuracy
# SPACY_MODEL = 'en_core_web_sm'
```

---

## 📝 Troubleshooting

### Common Issues

1. **Scrapy Crawling Issues**
   - Check that the target websites are accessible
   - Review `scrapy.log` for specific error messages
   - Adjust concurrent request settings in `settings.py`

2. **SpaCy GPU Issues**
   - Verify GPU availability with `nvidia-smi`
   - Make sure the SpaCy model is installed correctly
   - Try a smaller model if you encounter memory errors

3. **GPU Environment Issues**
   - Ensure GPU drivers are properly installed
   - Verify `genv` is correctly configured
   - Try running with `--gpu-count 0` to use CPU only

### Logging

The scraper outputs detailed logs to help with debugging:
- Spider activity in `scrapy.log`
- General execution in `scraper.log`
- HTTP caching in the `httpcache` directory

---

## 🔄 Maintenance and Updates

### Updating for Website Changes

If a website's structure changes:

1. Identify the new CSS selectors using browser developer tools
2. Update the corresponding parser method in `politician_spider.py`
3. Consider adding fallback selectors for more robust parsing

### Performance Tuning

Adjust the following settings in `settings.py` for better performance:

```python3
# Increase for more aggressive crawling (may trigger rate limits)
CONCURRENT_REQUESTS = 16

# Decrease to reduce load on target websites
DOWNLOAD_DELAY = 1.5

# For GPU processing, adjust batch size based on available memory
# In pipelines.py, adjust self.nlp.batch_size
```

### Adding New Entity Types

To extract additional entity types:

1. Add new fields to `PoliticianItem` in `items.py`
2. Update the `SpacyNERPipeline` in `pipelines.py` to extract and populate these fields
3. Customize entity extraction logic based on your specific needs