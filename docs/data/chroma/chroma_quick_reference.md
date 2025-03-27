# ChromaDB Quick Reference Guide

This document provides a quick reference of common commands for interacting with the ChromaDB database at `/home/natalie/political_db`.

## Setup & Initialization

**Initialize database**
```bash
cd src/data/db/chroma
./setup.sh
```

## Basic Commands

**View all politicians**
```bash
python3 docs/query_database.py --list-all
```

**Search by topic (semantic search)**
```bash
python3 docs/query_database.py --query "climate change" --results 5
```

**Get detailed information about a politician**
```bash
python3 docs/query_database.py --id "<politician-id>"
```

**Get detailed list of all politicians**
```bash
python3 docs/query_database.py --list-all --detailed
```

## python3 Code Snippets

**Connect to database**
```python3
from src.data.db.chroma.schema import connect_to_chroma, get_collection

client = connect_to_chroma("/home/natalie/political_db")
collection = get_collection(client)
```

**Search politicians**
```python3
from src.data.db.chroma.operations import search_politicians

results = search_politicians(collection, "foreign policy", n_results=3)
```

**Get all politicians**
```python3
from src.data.db.chroma.operations import get_all_politicians

all_politicians = get_all_politicians(collection)
```

**Add a politician**
```python3
from src.data.db.chroma.operations import add_politician

politician_data = {
    "name": "Jane Doe",
    "political_affiliation": "Independent",
    "biography": "Jane Doe is a fictional politician...",
    "policies": '{"economy": ["Support small businesses"]}'
}

add_politician(collection, politician_data)
```

## Database Management

**Create backup**
```bash
tar -czvf political_db_backup_$(date +%Y%m%d).tar.gz /home/natalie/political_db
```

**Check permissions**
```bash
ls -la /home/natalie/political_db
```

**Fix permissions**
```bash
sudo chown -R username:username /home/natalie/political_db
sudo chmod -R 755 /home/natalie/political_db
```

## Troubleshooting

**Check database status**
```python3
import chromadb
client = chromadb.PersistentClient(path="/home/natalie/political_db")
print(client.list_collections())
```

**Clear query cache (performance)**
```bash
rm -rf /home/natalie/political_db/chroma.sqlite3-shm /home/natalie/political_db/chroma.sqlite3-wal
```

For more detailed instructions, see the [full database guide](./chroma_database_guide.md). 