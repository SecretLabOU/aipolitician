# Data Pipeline Documentation

The AI Politician system includes a data processing pipeline that transforms raw scraped data into a format suitable for the knowledge database. This document explains how the pipeline works and how to use it.

## Overview

The data pipeline performs the following operations:
1. Loads raw data from the scraper
2. Cleans and normalizes the text
3. Chunks text into appropriate sizes
4. Extracts metadata and key information
5. Generates embeddings
6. Prepares data for database insertion

## Pipeline Components

The pipeline is implemented in `src/data/pipeline/pipeline.py` and consists of several key components:

1. **Data Loader**: Loads raw JSON files produced by the scraper
2. **Text Processor**: Cleans and normalizes text content
3. **Chunker**: Splits text into appropriate chunks for retrieval
4. **Metadata Extractor**: Extracts and enhances metadata
5. **Embedder**: Generates vector embeddings for text chunks
6. **Database Formatter**: Prepares data for insertion into Milvus

## Usage

### Running the Pipeline

To process data for a specific politician:

```bash
python -m src.data.pipeline.pipeline --input-dir ./data/raw/biden --output-dir ./data/processed/biden --politician biden
```

### Configuration Options

The pipeline supports several configuration options:

- `--input-dir`: Directory containing raw data files
- `--output-dir`: Directory to store processed data
- `--politician`: Target politician (biden, trump)
- `--chunk-size`: Size of text chunks in characters
- `--chunk-overlap`: Overlap between chunks in characters
- `--embedding-model`: Name of the embedding model to use
- `--batch-size`: Number of items to process in each batch
- `--metadata-extraction`: Enable or disable enhanced metadata extraction
- `--verbose`: Enable verbose logging

### Example Commands

```bash
# Process Biden data with default settings
python -m src.data.pipeline.pipeline --input-dir ./data/raw/biden --output-dir ./data/processed/biden --politician biden

# Process Trump data with custom chunk size
python -m src.data.pipeline.pipeline --input-dir ./data/raw/trump --output-dir ./data/processed/trump --politician trump --chunk-size 1000 --chunk-overlap 200
```

## Pipeline Stages

### 1. Data Loading

Reads raw JSON files produced by the scraper and validates their format.

### 2. Text Processing

- Removes HTML/markdown formatting
- Normalizes whitespace
- Corrects common OCR errors
- Standardizes quotes, apostrophes, and other punctuation
- Handles abbreviations consistently

### 3. Chunking

Splits text into chunks suitable for retrieval with options for:
- Fixed-size chunking
- Semantic chunking (splitting at logical boundaries)
- Recursive chunking for hierarchical documents

### 4. Metadata Extraction and Enhancement

- Extracts dates, locations, and other factual information
- Identifies policy areas and topics
- Maps to standardized categories
- Generates summary information

### 5. Embedding Generation

- Creates vector embeddings for text chunks
- Supports multiple embedding models
- Batches operations for efficiency

### 6. Database Preparation

- Formats data for Milvus insertion
- Creates unique IDs
- Validates against database schema

## Output Format

The pipeline outputs processed data in a format ready for database loading:

```json
{
  "id": "unique_identifier",
  "text": "Processed text chunk",
  "metadata": {
    "politician": "biden",
    "date": "2021-01-20",
    "source": "original_source_url",
    "type": "speech",
    "topics": ["economy", "healthcare"],
    "policy_areas": ["economic_policy"],
    "location": "Washington DC",
    "additional_metadata": "..."
  },
  "embedding": [0.123, 0.456, ...] // Vector embedding
}
```

## Requirements

The pipeline requires several libraries listed in `requirements/requirements-langgraph.txt`, including:

- numpy
- pandas
- sentence-transformers
- nltk
- spacy

Install these dependencies using:

```bash
pip install -r requirements/requirements-langgraph.txt
```

## Integration with Other Components

The pipeline connects to other system components in the following ways:

1. **Input**: Takes data from the scraper system
2. **Output**: Produces data for the database system
3. **Models**: Uses embedding models configured in the models component 