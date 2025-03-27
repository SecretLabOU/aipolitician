# ChromaDB for AI Politician

This directory contains the ChromaDB implementation for storing and retrieving political figure data used in the AI Politician debate system.

## Overview

ChromaDB is used to store text data and embeddings for political figures, enabling semantic search capabilities for the RAG (Retrieval Augmented Generation) system. The implementation uses the BGE-Small-EN model from Hugging Face for generating embeddings.

## Directory Structure

```
chroma/
├── __init__.py          # Package initialization
├── schema.py            # Database schema and connection functions
├── operations.py        # Database operations (upsert, query, etc.)
├── setup.sh             # Setup script for database initialization
└── README.md            # This file
```

## Setup

To set up the ChromaDB database, follow these steps:

1. Run the setup script:

```bash
cd src/data/db/chroma
chmod +x setup.sh
./setup.sh
```

This will:
- Install necessary dependencies (ChromaDB, sentence-transformers, etc.)
- Create the database directory with proper permissions
- Initialize the database

By default, the database is stored in `~/political_db`. This can be changed in the setup script or by passing a different path to the pipeline.

## Usage

### Running the Pipeline

The pipeline script will scrape data for political figures and store it in the ChromaDB database:

```bash
python src/data/pipeline/pipeline.py --politicians "Politician1,Politician2,..."
```

Or with a file containing politician names (one per line):

```bash
python src/data/pipeline/pipeline.py --file politicians.txt
```

To specify a custom database path:

```bash
python src/data/pipeline/pipeline.py --politicians "Politician1" --db-path /path/to/db
```

### Searching the Database

You can search the database programmatically using the operations module:

```python
from data.db.chroma.schema import connect_to_chroma, get_collection
from data.db.chroma.operations import search_politicians

# Connect to the database
client = connect_to_chroma()
collection = get_collection(client)

# Search for politicians
results = search_politicians(collection, "climate change policy", n_results=5)
```

## Permissions

The database directory is set up with the following permissions:
- Owner (you): Read, Write, Execute (rwx)
- Others: Read, Execute (r-x)

This ensures that:
- You can modify the database
- Other users can read from the database but cannot modify it

## Important Notes

1. The first time you run a search or insertion, it may take some time to download the BGE-Small-EN model from Hugging Face.
2. Make sure the directory permissions are maintained if you move or copy the database.
3. For production use, consider backing up the database regularly.

## Troubleshooting

If you encounter issues:

1. Check the logs in `political_pipeline.log`
2. Ensure all dependencies are installed correctly
3. Verify directory permissions with `ls -la ~/political_db`
4. Make sure you have enough disk space for the database
5. Check that the Python environment has access to the required packages 