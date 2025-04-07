# ChromaDB Setup Instructions

This document provides instructions for setting up ChromaDB for the AI Politician project and removing any old Milvus database references.

## 1. Clean Up Milvus (Previous Database)

If you previously used the Milvus database, you should remove it completely since the system now uses ChromaDB exclusively:

```bash
# Stop any running Milvus containers if they exist
docker stop $(docker ps -q --filter "name=milvus")

# Remove the src/data/db/milvus directory
rm -rf src/data/db/milvus

# Remove any Milvus containers (optional)
docker rm $(docker ps -a -q --filter "name=milvus")

# Remove Milvus images (optional)
docker rmi milvusdb/milvus:latest
```

## 2. Install ChromaDB Dependencies

Install the required Python packages for ChromaDB:

```bash
pip install chromadb==0.4.6 sentence-transformers==2.2.2 python-dotenv
```

ChromaDB requires the following key dependencies:
- `chromadb`: The vector database itself
- `sentence-transformers`: For creating embeddings
- `pydantic<2.0.0`: ChromaDB 0.4.6 requires Pydantic v1
- `python-dotenv`: For loading environment variables (optional)

## 3. Initialize ChromaDB

Initialize the ChromaDB database:

```bash
cd src/data/db/chroma
chmod +x setup.sh
./setup.sh
```

This script will:
1. Create a database directory at `~/political_db` (default)
2. Install all necessary dependencies
3. Initialize the ChromaDB database with the proper schema

No API keys are required for ChromaDB setup - it works completely locally.

## 4. Verify Installation

To verify the ChromaDB installation:

```bash
python check_system.py
```

You should see confirmation that:
- ChromaDB dependencies are installed
- The database is accessible

## 5. Load Sample Data

To load sample politician data into the database:

```bash
cd src/data/db/chroma
python loader.py --source-dir /path/to/json/files
```

## 6. Test RAG Functionality

After setup, test the RAG (Retrieval-Augmented Generation) functionality:

```bash
python langgraph_politician.py debate run --topic "Economy" --participants "trump,biden"
```

You should no longer see the warning about "RAG dependencies not found" if everything is set up correctly.

## Database Location

By default, the ChromaDB data is stored at:
- `~/political_db`

You can change this by modifying `DEFAULT_DB_PATH` in `src/data/db/chroma/schema.py`.

## Troubleshooting

If you encounter issues:

1. Check that the database directory exists and has correct permissions
2. Make sure all dependencies are installed
3. Verify the Python environment is correct
4. Check the logs in `chroma_loader.log` for any database loading errors 