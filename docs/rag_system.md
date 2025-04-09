# RAG System for AI Politician

This document explains how to set up and use the Retrieval-Augmented Generation (RAG) system for the AI Politician project.

## Overview

The RAG system enhances the AI response quality by retrieving relevant factual information from a database of political content before generating responses. This helps make the AI politicians' responses more accurate and grounded in facts.

## System Components

- **ChromaDB**: Vector database for storing and retrieving document embeddings
- **SentenceTransformer**: Model that creates embeddings (vector representations) of text
- **RAG Utilities**: Python code that integrates the database with the chat system

## Database Configuration

- **Location**: `/opt/chroma_db` (persistent storage)
- **Collection**: `politicians`
- **Client**: `PersistentClient` with SQLite backend
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional vectors)

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements-rag.txt
```

2. Ensure the ChromaDB database is accessible at `/opt/chroma_db`. If it's not present, you'll need to:
   - Create the directory: `sudo mkdir -p /opt/chroma_db`
   - Set appropriate permissions: `sudo chown $USER:$USER /opt/chroma_db`
   - Initialize the database (see section below)

## Usage

The RAG system is automatically used when chatting with the AI Politicians when the `--no-rag` flag is NOT specified:

```bash
# With RAG enabled (default)
python aipolitician.py chat biden         
python aipolitician.py debug biden
python aipolitician.py trace biden

# Without RAG
python aipolitician.py chat biden --no-rag
```

## How It Works

1. When a user asks a question, the system extracts the query text.
2. The query is embedded into a vector using the SentenceTransformer model.
3. ChromaDB searches for similar vectors in the database, filtered by politician name.
4. Relevant documents are retrieved and formatted into a context string.
5. This context is prepended to the model prompt, providing factual information.
6. The LLM generates a response using both the user query and the retrieved context.

## Troubleshooting

When running the application, you'll see detailed messages if there are issues with the RAG system. Here are common problems and solutions:

### Common Issues

1. **"ChromaDB not installed"**:
   - Solution: Run `pip install -r requirements-rag.txt`

2. **"SentenceTransformer not installed"**:
   - Solution: Run `pip install -r requirements-rag.txt`

3. **"Database path does not exist"**:
   - Solution: Create the directory and set permissions:
     ```
     sudo mkdir -p /opt/chroma_db
     sudo chown $USER:$USER /opt/chroma_db
     ```

4. **"Politicians collection not found in database"**:
   - Solution: The database exists but doesn't have the required collection. You need to initialize the database with politician data.

5. **"Failed to generate embeddings"**:
   - Solution: There might be issues with the embedding model. Check your internet connection as it may need to download the model.

### Verifying RAG is Working

When RAG is working correctly, you'll see this message when starting the application:
```
RAG database system available and operational.
```

In the chat, you'll notice that responses are more factually grounded and may reference specific sources.

### Running Without RAG

If you're having issues with the RAG system, you can always run the application with the `--no-rag` flag:
```
python aipolitician.py chat biden --no-rag
```

This will run the system with synthetic responses instead of retrieving information from the database. 