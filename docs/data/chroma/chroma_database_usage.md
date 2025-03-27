# ChromaDB Database Usage Guide

This guide provides instructions for working with the ChromaDB vector database used in the AI Politician project.

## Database Information

- **Database Location**: `/home/natalie/political_db`
- **Collection Name**: `political_figures`
- **Embedding Model**: BGE-Small-EN (BAAI/bge-small-en)
- **Vector Dimension**: 384

## Database Setup

If you need to set up the database from scratch, run:

```bash
cd src/data/db/chroma
./setup.sh
```

This will initialize an empty database with the correct permissions:
- Owner (you): Read, Write, Execute (rwx)
- Other users: Read, Execute (r-x)

## Viewing Database Contents

### List All Politicians in the Database

```bash
cd src/data/db/chroma
python3 search.py --list-all
```

### Search for Politicians by Topic

```bash
cd src/data/db/chroma
python3 search.py --query "climate change policy" --results 5
```

For more detailed results:

```bash
python3 search.py --query "healthcare" --results 3 --verbose
```

### Use Custom Database Path

If the database is in a different location:

```bash
python3 search.py --query "foreign policy" --db-path /path/to/db
```

## Analyzing Database Contents

For a quick look at database metadata:

```python3
# Run in python3
import sys
sys.path.append("/home/natalie/code/aipolitician")
from src.data.db.chroma.schema import connect_to_chroma, get_collection

# Connect to ChromaDB
client = connect_to_chroma("/home/natalie/political_db")
collection = get_collection(client)

# Get collection info
print(f"Collection: {collection.name}")
print(f"Count: {collection.count()}")
```

## Querying the Database from python3

Here's how to use the database in your own code:

```python3
import sys
sys.path.append("/home/natalie/code/aipolitician")
from src.data.db.chroma.schema import connect_to_chroma, get_collection
from src.data.db.chroma.operations import search_politicians, get_politician_by_id

# Connect to database
client = connect_to_chroma("/home/natalie/political_db")
collection = get_collection(client)

# Search by semantic query (returns similar results)
results = search_politicians(collection, "policy on renewable energy", n_results=5)
for i, result in enumerate(results, 1):
    print(f"{i}. {result.get('name')} - {result.get('political_affiliation')}")
    
# Get politician by ID
if results:
    politician_id = results[0]["id"]
    politician = get_politician_by_id(collection, politician_id)
    print(f"Details for {politician['name']}:")
    print(f"Biography: {politician['biography']}")
```

## Populating the Database

To add politicians to the database, use the pipeline:

```bash
cd /home/natalie/code/aipolitician
python3 src/data/pipeline/pipeline.py --politicians "Abraham Lincoln,Barack Obama,Donald Trump"
```

Or use a file with politician names (one per line):

```bash
python3 src/data/pipeline/pipeline.py --file politicians.txt --stats-output stats.json
```

## Backup and Export

To backup the database:

```bash
# Create a backup tarball
cd /home/natalie
tar -czvf political_db_backup_$(date +%Y%m%d).tar.gz political_db/
```

## Troubleshooting

If you encounter permission issues:

```bash
# Fix permissions
sudo chmod -R 755 /home/natalie/political_db
sudo chown -R natalie:natalie /home/natalie/political_db
```

If the database is corrupted:

```bash
# Reinitialize database
cd src/data/db/chroma
./setup.sh
```

For detailed logs:

```bash
# View pipeline logs
cat political_pipeline.log
```

## Advanced: Direct ChromaDB CLI Access

For advanced usage, you can interact directly with ChromaDB using python3:

```python3
import chromadb
from chromadb.config import Settings

# Connect directly
client = chromadb.PersistentClient(path="/home/natalie/political_db")

# Get collection
collection = client.get_collection("political_figures")

# View raw data
raw_data = collection.get(limit=5)
print(f"IDs: {raw_data['ids']}")
print(f"Metadata: {raw_data['metadatas']}")
``` 