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

# First install compatible versions of requirements
echo -e "${YELLOW}Installing compatible dependencies...${NC}"
pip install 'numpy<2.0.0'  # Use NumPy 1.x for compatibility
pip install 'pydantic<2.0.0'  # Install pydantic v1 which ChromaDB 0.3.26 expects

# Install required Python packages with specific versions
echo -e "${YELLOW}Installing Python dependencies...${NC}"
pip install 'chromadb==0.4.6'  # This version works with pydantic v1

# Install HuggingFace Transformers with specific versions for BGE model
echo -e "${YELLOW}Installing BGE embedding model dependencies...${NC}"
pip install 'transformers>=4.30.0' 'torch>=2.0.0'

# Download the BGE model to cache before using
echo -e "${YELLOW}Downloading BGE-Small-EN model...${NC}"
python -c "
from transformers import AutoTokenizer, AutoModel
import torch
# Check for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
# Download model and tokenizer files to cache
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-small-en')
model = AutoModel.from_pretrained('BAAI/bge-small-en').to(device)
print('Successfully downloaded BGE-Small-EN model and tokenizer')
"

# Completely remove the existing database directory and create a fresh one
echo -e "${YELLOW}Setting up a completely fresh database directory...${NC}"
if [ -d "$DB_DIR" ]; then
    echo -e "${YELLOW}Removing existing database directory at $DB_DIR${NC}"
    rm -rf "$DB_DIR"
fi

# Create a new clean directory
echo -e "${YELLOW}Creating fresh database directory at $DB_DIR...${NC}"
mkdir -p "$DB_DIR"

# Set permissions (rwxr-xr-x)
echo -e "${YELLOW}Setting directory permissions...${NC}"
chmod 755 "$DB_DIR"

echo -e "${GREEN}Verifying Python installation...${NC}"
python -c "
import numpy
import pydantic
import chromadb
import transformers
import torch
print(f'NumPy version: {numpy.__version__}')
print(f'Pydantic version: {pydantic.__version__}')
print(f'ChromaDB version: {chromadb.__version__}')
print(f'Torch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
print('All dependencies successfully installed')
"

# Initialize the database
echo -e "${YELLOW}Initializing the database...${NC}"
python -c "
import sys
import os
import shutil
import time

# Add the src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('$0'), '../../../')))

# Make sure directory is clean
db_path = '$DB_DIR'
if os.path.exists(db_path):
    print(f'Ensuring directory {db_path} is ready for use...')
    # Make sure we have proper permissions
    os.chmod(db_path, 0o755)

# Import our database initialization code
from data.db.chroma.schema import initialize_database

# Initialize the database
db = initialize_database(db_path)
print(f\"Database initialized successfully at {os.path.abspath(db_path)}\")
"

echo -e "${GREEN}ChromaDB setup complete!${NC}"
echo -e "${YELLOW}Database location: $DB_DIR${NC}"
echo -e "${YELLOW}You can now run the pipeline to populate the database.${NC}" 