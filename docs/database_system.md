# Database System Documentation

The AI Politician system uses a vector database (Milvus) to store and retrieve political knowledge. This document explains how the database system works and how to manage it.

## Overview

The system uses Milvus, a vector database, to provide Retrieval-Augmented Generation (RAG) capabilities. The database stores:

- Politician statements and speeches
- Policy positions
- Biographical information
- Historical political data

Each piece of information is converted into vector embeddings for semantic search and retrieval.

## Database Structure

The database is organized as follows:

- **Collection**: `political_knowledge` (default) or politician-specific collections
- **Schema**: Each document contains text content, metadata, and vector embeddings
- **Partition**: Data may be partitioned by politician name, date, or topic

## Connection Configuration

Database connection settings are managed through environment variables:

- `MILVUS_HOST`: Hostname of the Milvus server (default: localhost)
- `MILVUS_PORT`: Port of the Milvus server (default: 19530)
- `MILVUS_ALIAS`: Connection alias (default: default)
- `MILVUS_COLLECTION`: Collection name (default: political_knowledge)
- `MILVUS_COLLECTION_BIDEN`: Biden-specific collection (optional)
- `MILVUS_COLLECTION_TRUMP`: Trump-specific collection (optional)

These settings can be defined in the `.env` file.

## Managing the Database

The system includes scripts for managing the database:

### Loading Data

Use the `scripts/load_milvus_data.py` script to load processed data into the database:

```bash
python scripts/load_milvus_data.py --path path/to/processed/data --politician biden
```

### Database Management

Use the `scripts/manage_db.py` script for database management tasks:

```bash
# Create a new collection
python scripts/manage_db.py create --collection political_knowledge

# List all collections
python scripts/manage_db.py list

# Delete a collection
python scripts/manage_db.py drop --collection political_knowledge

# Count entries in a collection
python scripts/manage_db.py count --collection political_knowledge
```

## Database Usage in the System

The database is integrated into the AI Politician system as follows:

1. When a user query is received, the Context Agent extracts key topics and policy areas.
2. These topics are used to query the Milvus database for relevant knowledge.
3. Retrieved knowledge is added to the context for the Response Agent.
4. The Response Agent uses this knowledge to generate informed responses.

## Disabling the Database

The database/RAG system can be disabled using the `--no-rag` flag:

```bash
python aipolitician.py chat biden --no-rag
```

When disabled, the system will fall back to using the base language model without retrieved knowledge.

## Setup and Requirements

To use the database system, you need:

1. A running Milvus instance (local or remote)
2. Proper environment configuration
3. The Python dependencies listed in `requirements/requirements-langgraph.txt`

You can start a local Milvus instance using Docker:

```bash
docker run -d --name milvus_standalone -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest standalone
``` 