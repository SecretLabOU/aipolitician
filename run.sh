#!/bin/bash

# Check if .env exists
if [ ! -f .env ]; then
    echo "Error: .env file not found. Please create one from .env.example"
    exit 1
fi

# Environment setup
ENV_NAME="aipolitician"
REQUIREMENTS="requirements.txt"

# Create/update conda environment
if ! conda env list | grep -q $ENV_NAME; then
    echo "Creating conda environment..."
    conda create -n $ENV_NAME python=3.10 -y
fi

# Activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# Install requirements
pip install -r $REQUIREMENTS

# Load environment variables
set -a
source .env
set +a

# Run server
echo "Starting server..."
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

echo "Server is running. Access it at http://localhost:8000"
