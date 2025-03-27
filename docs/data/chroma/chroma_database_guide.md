# ChromaDB Database Guide

## Overview

This document provides comprehensive information for working with the AI Politician's ChromaDB vector database located at `/home/natalie/political_db`.

## Database Information

- **Location**: `/home/natalie/political_db`
- **Collection**: `political_figures`
- **Embedding Model**: BGE-Small-EN (HuggingFace Transformers)
- **Vector Dimension**: 384
- **Access Permissions**: 
  - Read access: All users
  - Write access: Database owner only

## Database Setup

To initialize the database from scratch:

```bash
cd src/data/db/chroma
./setup.sh
```

This will:
1. Create the database directory if it doesn't exist
2. Install required dependencies
3. Initialize the database with the correct schema
4. Set appropriate permissions

## Basic Database Operations

### Quick Access Script

We've provided a convenience script for common database operations:

```bash
python3 docs/query_database.py --help
```

### List All Politicians

To view all politicians in the database:

```bash
python3 docs/query_database.py --list-all
```

For detailed information on each politician:

```bash
python3 docs/query_database.py --list-all --detailed
```

### Search by Topic

To search politicians by a topic or query:

```bash
python3 docs/query_database.py --query "climate change policy" --results 5
```

### Get Politician by ID

To retrieve a specific politician by their ID:

```bash
python3 docs/query_database.py --id "<politician-id>"
```

## Programmatic Access

### Connecting to the Database

```python3
from src.data.db.chroma.schema import connect_to_chroma, get_collection

# Connect to the ChromaDB client
client = connect_to_chroma("/home/natalie/political_db")

# Get the political figures collection
collection = get_collection(client)
```

### Querying the Database

```python3
from src.data.db.chroma.operations import search_politicians

# Search for politicians by topic
results = search_politicians(collection, "climate change", n_results=5)

# Print results
for politician in results:
    print(f"Name: {politician.get('name')}")
    print(f"Affiliation: {politician.get('political_affiliation')}")
    print(f"Biography: {politician.get('biography')}")
    print("-" * 50)
```

### Get All Politicians

```python3
from src.data.db.chroma.operations import get_all_politicians

# Get all politicians
all_politicians = get_all_politicians(collection)
```

### Get Politician by ID

```python3
from src.data.db.chroma.operations import get_politician_by_id

# Get a specific politician by ID
politician = get_politician_by_id(collection, "politician-id")
```

## Adding Data to the Database

### Using the Pipeline

To add politicians through the data pipeline:

```bash
# From project root
python3 src/pipeline.py --db-path /home/natalie/political_db
```

### Adding a Single Politician

```python3
from src.data.db.chroma.operations import add_politician
from src.data.db.chroma.schema import connect_to_chroma, get_collection

# Connect to database
client = connect_to_chroma("/home/natalie/political_db")
collection = get_collection(client)

# Create politician data
politician_data = {
    "name": "Jane Doe",
    "political_affiliation": "Independent",
    "biography": "Jane Doe is a fictional politician...",
    "policies": '{"economy": ["Support small businesses"], "environment": ["Invest in renewable energy"]}'
}

# Add to database
add_politician(collection, politician_data)
```

## Database Maintenance

### Backup

To create a backup of the database:

```bash
# Create a tarball backup
tar -czvf political_db_backup_$(date +%Y%m%d).tar.gz /home/natalie/political_db
```

### Performance Optimization

ChromaDB uses memory-mapped files and benefits from having adequate RAM. For optimal performance:

- Ensure at least 8GB of RAM is available
- If using GPU acceleration, CUDA or ROCm support is recommended

## Troubleshooting

### Permission Issues

If you encounter permission issues:

```bash
# Check permissions
ls -la /home/natalie/political_db

# Fix permissions if needed (replace username with actual user)
sudo chown -R username:username /home/natalie/political_db
sudo chmod -R 755 /home/natalie/political_db
```

### Database Corruption

If the database becomes corrupted:

1. Create a backup of the current state
2. Reinitialize the database using the setup script
3. Restore data from a pipeline or other source

### Import/Export

For sharing or transferring data:

```python3
from src.data.db.chroma.operations import export_politicians, import_politicians

# Export all politicians to a JSON file
export_politicians(collection, "export.json")

# Import politicians from a JSON file
import_politicians(collection, "export.json")
```

## Advanced: Direct ChromaDB Interaction

For advanced use cases, you can interact directly with the ChromaDB client:

```python3
import chromadb

# Connect directly
client = chromadb.PersistentClient(path="/home/natalie/political_db")
collection = client.get_collection("political_figures")

# Raw query
result = collection.query(
    query_texts=["climate change"],
    n_results=5
)

# Access raw embeddings
embeddings = collection.get(
    include=[
        "embeddings",
        "documents",
        "metadatas"
    ]
)
```

## Additional Resources

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Hugging Face BGE-Small-EN Model](https://huggingface.co/BAAI/bge-small-en)
- [Vector Database Concepts](https://www.pinecone.io/learn/vector-database/) 