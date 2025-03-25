# Usage Guide

This guide provides instructions for using the various components of the AI Politician system.

## Table of Contents

- [Chat System](#chat-system)
- [Database Management](#database-management)
- [Data Scraping](#data-scraping)
- [Data Pipeline](#data-pipeline)
- [Model Training](#model-training)
- [System Integration](#system-integration)

## Chat System

The chat system allows you to interact with AI versions of politicians.

### Basic Chat

```bash
# Chat with Biden
python aipolitician.py chat biden

# Chat with Trump
python aipolitician.py chat trump
```

### Debug Mode

To see additional debugging information while chatting:

```bash
python aipolitician.py debug biden
```

This shows sentiment analysis, knowledge retrieval status, and deflection reasoning.

### Trace Mode

For developers who need to see detailed workflow information:

```bash
python aipolitician.py trace biden
```

This displays detailed information about each step in the workflow.

### Disabling Knowledge Retrieval

To run without the knowledge database (RAG):

```bash
python aipolitician.py chat biden --no-rag
```

### Direct API Access

For programmatic use, you can process a single input:

```bash
python langgraph_politician.py process --identity biden --input "What's your position on climate change?"
```

This returns a JSON response that includes the generated text and metadata.

## Database Management

The system uses a Milvus vector database for knowledge retrieval.

### Collection Management

```bash
# Create a new collection
python scripts/database/manage_db.py create --collection political_knowledge

# List all collections
python scripts/database/manage_db.py list

# Drop a collection
python scripts/database/manage_db.py drop --collection political_knowledge

# Get collection statistics
python scripts/database/manage_db.py stats --collection political_knowledge
```

### Loading Data

To load processed data into the database:

```bash
python scripts/database/load_milvus_data.py --path data/processed/biden --politician biden
```

## Data Scraping

The scraper collects data about politicians from various sources.

### Basic Scraping

```bash
python -m src.data.scraper.politician_scraper --politician biden --output-dir ./data/raw/biden
```

### Advanced Scraping Options

```bash
# Scrape with date range
python -m src.data.scraper.politician_scraper --politician biden --output-dir ./data/raw/biden --start-date 2021-01-20 --end-date 2023-01-20

# Scrape specific sources
python -m src.data.scraper.politician_scraper --politician trump --output-dir ./data/raw/trump --sources whitehouse,campaign,twitter

# Set maximum items per source
python -m src.data.scraper.politician_scraper --politician biden --output-dir ./data/raw/biden --max-items 100
```

## Data Pipeline

The pipeline processes raw scraped data into a format suitable for the database.

### Running the Pipeline

```bash
python -m src.data.pipeline.pipeline --input-dir ./data/raw/biden --output-dir ./data/processed/biden --politician biden
```

### Customizing Chunking

```bash
python -m src.data.pipeline.pipeline --input-dir ./data/raw/biden --output-dir ./data/processed/biden --politician biden --chunk-size 1000 --chunk-overlap 200
```

### Selecting Embedding Model

```bash
python -m src.data.pipeline.pipeline --input-dir ./data/raw/biden --output-dir ./data/processed/biden --politician biden --embedding-model sentence-transformers/all-mpnet-base-v2
```

## Model Training

The training system fine-tunes models to capture politician speaking styles and policy positions.

### Training a New Model

```bash
python -m src.models.training.train_identity_adapter \
  --base-model mistralai/Mistral-7B-v0.1 \
  --politician biden \
  --data-path ./data/training/biden \
  --output-dir ./models/biden \
  --epochs 3 \
  --batch-size 4
```

### Merging Adapters

```bash
python -m src.models.training.merge_adapters \
  --base-model mistralai/Mistral-7B-v0.1 \
  --adapters ./models/biden-style,./models/biden-policy \
  --weights 0.7,0.3 \
  --output-dir ./models/biden-merged
```

## System Integration

For a full end-to-end workflow:

1. **Scrape data**:
   ```bash
   python -m src.data.scraper.politician_scraper --politician biden --output-dir ./data/raw/biden
   ```

2. **Process data**:
   ```bash
   python -m src.data.pipeline.pipeline --input-dir ./data/raw/biden --output-dir ./data/processed/biden --politician biden
   ```

3. **Load data into database**:
   ```bash
   python scripts/database/load_milvus_data.py --path ./data/processed/biden --politician biden
   ```

4. **Chat with the AI politician**:
   ```bash
   python aipolitician.py chat biden
   ```

## Workflow Visualization

To visualize the LangGraph workflow:

```bash
python langgraph_politician.py visualize
```

This creates an HTML file showing the workflow graph structure and opens it in your browser.

## System Check

To verify all components are working properly:

```bash
python check_system.py
```

This script checks:
- Required packages
- Database connection
- Model availability
- Environment configuration 