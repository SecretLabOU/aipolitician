#!/bin/bash
# ChromaDB Setup Script for AI Politician Project
# This script sets up ChromaDB and installs all necessary dependencies

# Exit on any error
set -e

# Set the database directory
DB_DIR="$HOME/political_db"

# Color for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up ChromaDB for AI Politician project...${NC}"

# First install NumPy with pinned version (before ChromaDB)
echo -e "${YELLOW}Installing NumPy with compatible version...${NC}"
pip install 'numpy<2.0.0'  # Use NumPy 1.x which is compatible with ChromaDB 0.4.18

# Install required Python packages with specific versions
echo -e "${YELLOW}Installing Python dependencies...${NC}"
pip install 'chromadb==0.4.18'  # Pinning to specific version for stability

# Install HuggingFace Transformers with specific versions for BGE model
echo -e "${YELLOW}Installing BGE embedding model dependencies...${NC}"
pip install 'transformers>=4.30.0' 'torch>=2.0.0'

# Download the BGE model to cache before using
echo -e "${YELLOW}Downloading BGE-Small-EN model...${NC}"
python -c "
from transformers import AutoTokenizer, AutoModel
# Download model and tokenizer files to cache
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-small-en')
model = AutoModel.from_pretrained('BAAI/bge-small-en')
print('Successfully downloaded BGE-Small-EN model and tokenizer')
"

# Create the database directory if it doesn't exist
if [ ! -d "$DB_DIR" ]; then
    echo -e "${YELLOW}Creating database directory at $DB_DIR...${NC}"
    mkdir -p "$DB_DIR"
else
    echo -e "${YELLOW}Database directory already exists at $DB_DIR${NC}"
fi

# Set permissions (rwxr-xr-x)
echo -e "${YELLOW}Setting directory permissions...${NC}"
chmod 755 "$DB_DIR"

echo -e "${GREEN}Verifying Python installation...${NC}"
python -c "
import numpy
import chromadb
import transformers
import torch
print(f'NumPy version: {numpy.__version__}')
print(f'ChromaDB version: {chromadb.__version__}')
print('All dependencies successfully installed')
"

# Clear any existing collections to avoid compatibility issues
echo -e "${YELLOW}Cleaning up any existing collections...${NC}"
python -c "
import sys
import os
import chromadb
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('$0'), '../../../')))
from data.db.chroma.schema import connect_to_chroma, DEFAULT_DB_PATH, DEFAULT_COLLECTION_NAME
client = connect_to_chroma('$DB_DIR')
try:
    client.delete_collection(name=DEFAULT_COLLECTION_NAME)
    print(f'Deleted existing collection {DEFAULT_COLLECTION_NAME}')
except Exception as e:
    print(f'No existing collection to delete or error: {e}')
"

# Initialize the database
echo -e "${YELLOW}Initializing the database...${NC}"
python -c "
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('$0'), '../../../')))
from data.db.chroma.schema import initialize_database
db = initialize_database('$DB_DIR')
print(f\"Database initialized successfully at {os.path.abspath('$DB_DIR')}\")
"

echo -e "${GREEN}ChromaDB setup complete!${NC}"
echo -e "${YELLOW}Database location: $DB_DIR${NC}"
echo -e "${YELLOW}You can now run the pipeline to populate the database.${NC}" 