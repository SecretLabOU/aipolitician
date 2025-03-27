# AI Politician Data Pipeline Documentation

This document describes the data pipeline for the AI Politician project, which scrapes information about political figures, processes it, and stores it in a ChromaDB vector database.

## Pipeline Overview

The AI Politician data pipeline consists of several components:

1. **Data Scraping**: Collection of political figure data using the politician scraper
2. **Data Processing**: Transforming raw data into a suitable format for vector embeddings
3. **Vector Embedding**: Creating embeddings using the BGE-Small-EN model
4. **Database Storage**: Storing data in ChromaDB with proper indexing
5. **Reporting**: Generating statistics and logs about the pipeline run

## System Architecture

```
[Scraper] --> [Data Processor] --> [Vector Embedder] --> [ChromaDB]
    ^                                                       |
    |                                                       v
    +------------------[Pipeline Controller]--------------[Logs]
```

## Pipeline Components

### 1. Politician Scraper

The scraper collects data about political figures from various sources, including:
- Basic biographical information
- Political positions and affiliations
- Policy statements
- Legislative records
- Public communications

```python3
from src.data.scraper.politician_scraper import crawl_political_figure

# Scrape data for a specific politician
politician_data = await crawl_political_figure("Politician Name")
```

### 2. Data Processor

The processor transforms raw scraped data into a structured format suitable for embedding and storage:

```python3
from src.data.pipeline.pipeline import map_scraper_to_chroma

# Transform scraped data to ChromaDB format
chroma_data = map_scraper_to_chroma(scraped_data)
```

### 3. Vector Embedding

The system uses the BGE-Small-EN model to create embeddings of political content. The embeddings are 384-dimensional vectors that capture the semantic meaning of the text.

```python3
# The embedding happens automatically when inserting into ChromaDB
# through the custom BGEEmbeddingFunction class in schema.py
```

### 4. ChromaDB Storage

Data is stored in ChromaDB, a vector database optimized for similarity search:

```python3
from src.data.db.chroma.schema import connect_to_chroma, get_collection
from src.data.db.chroma.operations import upsert_politician

# Store data in ChromaDB
client = connect_to_chroma("/home/natalie/political_db")
collection = get_collection(client)
doc_id = upsert_politician(collection, politician_data)
```

## Running the Pipeline

The pipeline can be run in two ways:

### 1. Processing Individual Politicians

```bash
python3 src/data/pipeline/pipeline.py --politicians "Abraham Lincoln,Barack Obama,Donald Trump"
```

### 2. Processing a List of Politicians from a File

```bash
python3 src/data/pipeline/pipeline.py --file politicians.txt --stats-output stats.json
```

### Command-line Arguments

| Argument | Description |
|----------|-------------|
| `--politicians` | Comma-separated list of politician names to process |
| `--file` | Path to a file containing politician names (one per line) |
| `--stats-output` | Path to save pipeline statistics JSON (optional) |
| `--db-path` | Path to ChromaDB database (default: ~/political_db) |

## Database Configuration

The pipeline uses ChromaDB with the following configuration:

- **Database Location**: `/home/natalie/political_db`
- **Collection Name**: `political_figures`
- **Embedding Model**: BGE-Small-EN
- **ChromaDB Version**: 0.4.6 (with Pydantic v1)
- **GPU Acceleration**: Automatically uses available GPUs (preferring RTX 4090 if available)

## Pipeline Flow

1. The pipeline starts by connecting to ChromaDB
2. For each politician:
   - Scrapes data using the politician scraper
   - Maps the scraped data to the ChromaDB format
   - Creates embeddings using the BGE-Small-EN model
   - Stores the politician data in ChromaDB
3. Generates statistics about the pipeline run
4. Outputs logs and statistics

## Recent Changes

- **Migration from Milvus to ChromaDB**: The system now uses ChromaDB which is simpler to set up and manage
- **GPU Acceleration**: Added automatic GPU detection and utilization
- **Improved Error Handling**: Better error recovery and reporting
- **Permission Management**: Database files have controlled permissions (owner rwx, others r-x)
- **Compatibility Fixes**: Addressed version issues with ChromaDB and its dependencies

## Monitoring and Troubleshooting

### Logs

The pipeline outputs detailed logs at:
```
./political_pipeline.log
```

### Pipeline Statistics

After each run, the pipeline outputs statistics in JSON format:

```json
{
  "total": 10,
  "successful": 8,
  "failed": 2,
  "skipped": 0,
  "start_time": "2025-03-27T14:32:45.123456",
  "end_time": "2025-03-27T14:35:12.789012",
  "duration_seconds": 147.665556,
  "details": {
    "Abraham Lincoln": "success",
    "Barack Obama": "success",
    "Invalid Politician": "failed: No data found"
  }
}
```

### Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| Scraper timeout | Increase timeout settings in scraper configuration |
| Database connection error | Ensure ChromaDB is properly set up with `./setup.sh` |
| Embedding model error | Check GPU availability and memory |
| Permission denied | Check permissions on the database directory |
| Import errors | Ensure all dependencies are installed |

## Performance Considerations

- **GPU Acceleration**: Using a GPU significantly accelerates embedding generation
- **Batch Processing**: Processing politicians in batches is more efficient
- **Parallel Scraping**: The scraper uses asynchronous I/O for better performance

## Future Improvements

- Distributed processing for large-scale politician data collection
- Improved data validation and cleaning
- Integration with additional data sources
- Scheduled automated updates 