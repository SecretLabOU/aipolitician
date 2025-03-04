# Political RAG Database System

This directory contains the database system for the AI Politician RAG (Retrieval-Augmented Generation) system. The database system stores structured data about Donald Trump and Joe Biden that can be used to provide factual information to the AI models.

## Directory Structure

- `config.py`: Configuration for database file paths
- `database.py`: Base database interface
- `schemas/`: Schema definitions for each database
- `scripts/`: Scripts for initializing and populating databases
- `utils/`: Utility functions for API access, web scraping, and embeddings

## Databases

The system includes the following databases:

1. **Biography Database**: Personal and professional biographical details
2. **Policy Database**: Policy positions, voting history, and political milestones
3. **Voting Record Database**: Detailed voting history on key legislative bills
4. **Public Statements Database**: Speeches, interviews, and official statements
5. **Fact-Check Database**: Fact-checked claims with verdicts
6. **Timeline Database**: Significant personal and professional events chronologically
7. **Legislative Database**: Bills sponsored, co-sponsored, or supported
8. **Campaign Promises Database**: Promises made during campaigns and fulfillment status
9. **Executive Actions Database**: Executive orders, memoranda, and proclamations
10. **Media Coverage Database**: News articles and media appearances
11. **Public Opinion Database**: Polling data about public perception
12. **Controversies Database**: Scandals, controversies, and investigations
13. **Policy Comparison Database**: Side-by-side comparisons of positions on key issues
14. **Judicial Appointments Database**: Judicial nominations and confirmations
15. **Foreign Policy Database**: Foreign policy decisions and international engagements
16. **Economic Metrics Database**: Economic indicators during presidencies
17. **Charitable Work Database**: Charitable contributions and efforts

## Installation

The database files are stored outside the git repository to avoid bloating the repository with data files. The actual database files are stored in `/home/natalie/Databases/political_rag/`.

Make sure this directory exists before running any database scripts:

```bash
mkdir -p /home/natalie/Databases/political_rag
```

## Usage

### Initializing Databases

To initialize all databases and populate them with initial data:

```bash
python -m db.scripts.initialize_databases
```

### Using Databases in Code

```python
from db import get_database, get_embedding_index

# Get a specific database
biography_db = get_database('biography')

# Query the database
trump = biography_db.get_politician_by_name("Donald Trump")
trump_bio = biography_db.get_complete_biography(trump['id'])

# Search using embeddings
results = biography_db.search_biography("economic policy")
```

## Data Sources

The database system collects data from the following sources:

- Wikipedia API
- Congress.gov API
- The American Presidency Project
- FactCheck.org and PolitiFact
- Pew Research and Gallup
- OpenSecrets API
- U.S. Bureau of Economic Analysis

## Embedding and Retrieval

The system includes an embedding system that converts text into vector representations, enabling semantic search across the databases. This is a key component for the RAG system, allowing the AI models to retrieve relevant information based on the semantic meaning of queries.

## Future Improvements

- Add more data sources
- Implement more sophisticated data extraction techniques
- Add periodic data updating
- Enhance embedding models for better retrieval
- Implement cross-database querying