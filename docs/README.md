# AI Politician Documentation

This directory contains comprehensive documentation for the AI Politician system.

## Documentation Index

### Overview
- [System Overview](system_overview.md) - High-level overview of the system architecture

### Components
- [Chat System](chat_system.md) - Documentation for the chat interface
- [LangGraph Workflow](langgraph_workflow.md) - Documentation for the LangGraph workflow
- [Database System](database_system.md) - Documentation for the Milvus vector database
- [Scraper System](scraper_system.md) - Documentation for the data collection scraper
- [Data Pipeline](pipeline_system.md) - Documentation for the data processing pipeline
- [Model Training](model_training.md) - Documentation for model training and fine-tuning

### Guides
- [Installation Guide](installation.md) - Detailed installation instructions
- [Usage Guide](usage_guide.md) - How to use the system's various components

## Quick Installation

To install the AI Politician system:

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/ai-politician.git
   cd ai-politician
   ```

2. Set up the environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install base requirements
   ```bash
   pip install -r requirements/requirements-base.txt
   ```

4. Install component-specific requirements as needed
   ```bash
   # For all components
   pip install -r requirements/requirements-all.txt
   
   # Or for specific components:
   pip install -r requirements/requirements-chat.txt
   pip install -r requirements/requirements-langgraph.txt
   pip install -r requirements/requirements-scraper.txt
   pip install -r requirements/requirements-training.txt
   ```

5. Set up environment variables by copying and modifying the example
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

6. Set up the Milvus database (if using RAG)
   ```bash
   docker run -d --name milvus_standalone -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest standalone
   ```

## Quick Start

Start chatting with the AI politician:

```bash
# Chat with Biden
python aipolitician.py chat biden

# Chat with Trump
python aipolitician.py chat trump
```

See the [Usage Guide](usage_guide.md) for more detailed instructions on using all components.

# AI Politician Database Documentation

This directory contains comprehensive documentation for working with the AI Politician's ChromaDB vector database.

## Documentation Files

- [ChromaDB Database Guide](./chroma_database_guide.md) - Complete documentation for setting up, using, and maintaining the database
- [ChromaDB Quick Reference](./chroma_quick_reference.md) - Common commands and code snippets for quick reference
- [ChromaDB Schema Reference](./chroma_schema_reference.md) - Details about the database schema and data structure
- [query_database.py](./query_database.py) - Executable script for querying the database from the command line

## Database Location

The ChromaDB database is located at:
```
/home/natalie/political_db
```

## Getting Started

If you're new to this database, start with the following steps:

1. Review the [ChromaDB Database Guide](./chroma_database_guide.md) for a complete overview
2. Run `./query_database.py --list-all` to see the politicians in the database
3. Try searching with `./query_database.py --query "climate change"` to find relevant politicians

## Common Tasks

### Viewing Database Contents

```bash
# List all politicians
./query_database.py --list-all

# Get detailed information
./query_database.py --list-all --detailed
```

### Searching the Database

```bash
# Search by topic
./query_database.py --query "healthcare policy" --results 5

# Get politician by ID
./query_database.py --id "<politician-id>"
```

### Database Setup

```bash
# Initialize the database
cd ../src/data/db/chroma
./setup.sh
```

## Support

If you encounter issues with the database, refer to the troubleshooting section in the [ChromaDB Database Guide](./chroma_database_guide.md) or contact the database administrator. 