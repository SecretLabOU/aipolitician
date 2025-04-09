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

The RAG system is automatically used when chatting with the AI Politicians when the `--rag` flag is enabled (or when `--no-rag` is not specified).

```bash
python aipolitician.py chat biden          # With RAG enabled
python aipolitician.py chat biden --no-rag  # Without RAG
```

## How It Works

1. When a user asks a question, the system extracts the query text.
2. The query is embedded into a vector using the SentenceTransformer model.
3. ChromaDB searches for similar vectors in the database, filtered by politician name.
4. Relevant documents are retrieved and formatted into a context string.
5. This context is prepended to the model prompt, providing factual information.
6. The LLM generates a response using both the user query and the retrieved context.

## Troubleshooting

- If you see a warning about "RAG database system not available", ensure:
  - The dependencies are installed
  - The database path exists and is accessible
  - The `politicians` collection exists in the database

- If results don't seem relevant:
  - The default number of results is 5, which can be adjusted in the code
  - The query embeddings might not match well with the stored document embeddings
  - The collection might not have relevant documents for that query 