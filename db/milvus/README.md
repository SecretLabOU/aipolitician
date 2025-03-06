# Milvus Database for Political Figure Retrieval

## 1. Overview

This database is designed for real-time political figure retrieval to support AI debates. It leverages Milvus as a vector database to enable semantic search functionality for political information.

Key Milvus features used in this project:
- **Vector Similarity Search**: Find relevant political information using semantic similarity
- **Schema Flexibility**: Combine structured data with vector embeddings
- **HNSW Indexing**: High-performance approximate nearest neighbor search

## 2. Installation

### Docker Deployment

The easiest way to set up the environment is using the provided scripts:

```bash
# Set up Milvus and dependencies
./setup.sh

# To tear down the environment when done
./cleanup.sh
```

Alternatively, you can run Milvus directly with Docker:

```bash
docker run -d --name milvus -p 19530:19530 milvusdb/milvus:latest
```

Or use the provided docker-compose.yml file:

```bash
docker compose up -d
```

### Python Dependencies

```python
pymilvus==2.2.0
sentence-transformers==2.2.2
```

You can install the dependencies using:

```bash
pip install pymilvus==2.2.0 sentence-transformers==2.2.2
```

## 3. Database Schema

The political figures collection has the following schema:

| Field | Type | Description |
|---|---|---|
| id | VARCHAR(36) | Primary key, UUID |
| name | VARCHAR(255) | Political figure's name |
| date_of_birth | VARCHAR(10) | Birth date (YYYY-MM-DD) |
| nationality | VARCHAR(100) | Country of origin |
| political_affiliation | VARCHAR(100) | Party or political leaning |
| biography | VARCHAR(65535) | Full biography text |
| positions | JSON | Political positions held |
| policies | JSON | Political stance data |
| legislative_actions | JSON | Voting records and sponsored bills |
| public_communications | JSON | Speeches and public statements |
| timeline | JSON | Key events timeline |
| campaigns | JSON | Campaign information |
| media | JSON | Media appearances and links |
| philanthropy | JSON | Charitable activities |
| personal_details | JSON | Additional personal information |
| embedding | FLOAT_VECTOR(768) | all-MiniLM-L6-v2 output vector |

## 4. Script Usage

### Database Initialization

To initialize or reset the database:

```bash
python3 scripts/initialize_db.py --recreate
```

This script:
- Establishes connection to Milvus
- Creates the political_figures collection with the schema above
- Builds HNSW index on the embedding field

### Search API Usage

Example code for searching political figures:

```python
from scripts.search import search_political_figures

# Search for political figures based on a query
results = search_political_figures(
    query="What is the stance on climate change?",
    limit=5,
    output_fields=["name", "political_affiliation", "policies"]
)

# Print results
for result in results:
    print(f"Name: {result['name']}")
    print(f"Affiliation: {result['political_affiliation']}")
    print(f"Score: {result['score']}")
    print(f"Policies: {result['policies']}")
    print("---")
```

## 5. Embedding Strategy

This project uses the following embedding approach:

- **Model**: all-MiniLM-L6-v2 sentence transformer model
- **Vector Dimensions**: 768-dimensional float vectors
- **Index Type**: HNSW (Hierarchical Navigable Small World)
- **Index Parameters**:
    - M=16 (maximum number of connections per layer)
    - efConstruction=200 (size of dynamic candidate list during construction)
- **Search Parameters**:
    - ef=100 (size of the dynamic candidate list during search)
    - Metric: L2 (Euclidean distance)

The embedding model converts politician biographical text and policy statements into numerical vectors, enabling semantic similarity search.

## 6. Directory Structure

```
./
├── scripts/
│   ├── initialize_db.py   # Database initialization
│   ├── schema.py          # Collection definition
│   └── search.py          # Vector query logic
├── logs/                  # Application logs
├── docker-compose.yml     # Docker configuration
├── setup.sh               # Setup script for environment
├── cleanup.sh             # Cleanup script for environment
└── README.md              # This documentation
```

## 7. Troubleshooting

### Common Errors

**Milvus Connection Issues**
- Error: "Failed to connect to Milvus server"
- Solution: Ensure Milvus container is running with `docker ps` and check logs with `docker logs milvus`

**Index Creation Failures**
- Error: "Failed to create index"
- Solution: Verify sufficient memory is available for index creation
- Alternative: Adjust index parameters (M, efConstruction) for lower memory usage

**JSON Field Format Requirements**
- Error: "Invalid field value type"
- Solution: Ensure JSON fields contain valid JSON structures
- Example: `policies` field should use the format:
  ```json
  {
    "climate_change": {"position": "supportive", "details": "..."},
    "economy": {"position": "mixed", "details": "..."}
  }
  ```

### Performance Optimization

- Increase `M` parameter for better search accuracy (at the cost of memory)
- Decrease `efConstruction` for faster index building (at the cost of recall)
- Adjust `ef` search parameter based on query performance requirements
