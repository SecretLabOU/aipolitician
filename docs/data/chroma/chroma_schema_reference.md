# ChromaDB Schema Reference

This document outlines the structure and schema of the political figures database in ChromaDB located at `/home/natalie/political_db`.

## Database Structure

ChromaDB is a vector database that stores embeddings alongside documents and metadata. The database consists of:

- **Embeddings**: Vector representations of text content (384-dimensional vectors)
- **Documents**: Original text content that was embedded
- **Metadata**: Structured data associated with each embedding/document
- **IDs**: Unique identifiers for each entry

## Collection: `political_figures`

The database contains a single collection named `political_figures` that stores information about politicians.

## Schema

Each politician entry contains the following fields:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `id` | String | Unique identifier for the politician | `"john-doe-2023"` |
| `name` | String | Full name of the politician | `"John Doe"` |
| `political_affiliation` | String | Political party or affiliation | `"Democrat"` |
| `biography` | String | Biographical information | `"John Doe is a senator who..."` |
| `policies` | JSON String | JSON object containing policy positions by category | `'{"economy": ["Tax reform"], "healthcare": ["Universal coverage"]}'` |

### Metadata Structure

The metadata for each entry is structured as follows:

```json
{
  "id": "john-doe-2023",
  "name": "John Doe",
  "political_affiliation": "Democrat",
  "biography": "John Doe is a senator who has served for 10 years...",
  "policies": "{\"economy\": [\"Tax reform\", \"Job creation\"], \"healthcare\": [\"Universal coverage\"]}"
}
```

### Document Structure

The document field contains a concatenation of the politician's information in a format that's optimized for embedding and semantic search:

```
Name: John Doe
Political Affiliation: Democrat
Biography: John Doe is a senator who has served for 10 years...
Policies:
- Economy: Tax reform, Job creation
- Healthcare: Universal coverage
```

## Embedding Function

Politicians' information is embedded using the BGE-Small-EN model from HuggingFace Transformers. This model produces 384-dimensional embeddings that capture the semantic meaning of the text.

## Search Behavior

When you search the database:

1. Your query is converted to an embedding using the same model
2. ChromaDB performs a similarity search to find the closest embeddings
3. Results are returned sorted by relevance (cosine similarity)

## Example Entry

```python3
{
  "id": "john-doe-2023",
  "metadata": {
    "id": "john-doe-2023",
    "name": "John Doe",
    "political_affiliation": "Democrat",
    "biography": "John Doe is a senator who has served for 10 years...",
    "policies": "{\"economy\": [\"Tax reform\", \"Job creation\"], \"healthcare\": [\"Universal coverage\"]}"
  },
  "document": "Name: John Doe\nPolitical Affiliation: Democrat\nBiography: John Doe is a senator who has served for 10 years...\nPolicies:\n- Economy: Tax reform, Job creation\n- Healthcare: Universal coverage",
  "embedding": [0.123, 0.456, ..., 0.789]  # 384-dimensional vector (abbreviated)
}
```

## Database Implementation Details

The database uses ChromaDB's PersistentClient, which stores data in:

- SQLite files for metadata and document storage
- HNSW (Hierarchical Navigable Small World) indices for vector search

The database files are stored at `/home/natalie/political_db` with the following structure:

```
/home/natalie/political_db/
├── chroma.sqlite3      # Main database file
├── chroma.sqlite3-shm  # SQLite shared memory file
├── chroma.sqlite3-wal  # SQLite write-ahead log
└── index/              # Directory containing vector indices
    └── ...
``` 