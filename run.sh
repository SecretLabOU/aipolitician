#!/bin/bash

# Check if .env exists
if [ ! -f .env ]; then
    echo "Error: .env file not found. Please create one from .env.example"
    exit 1
fi

# Environment setup
ENV_NAME="mistral-finetune"  # Match the working environment name

# Activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# Install/update dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Load environment variables
set -a
source .env
set +a

# Activate GPU environment
echo "Setting up GPU..."
genv activate --id nat
genv attach --index 0

# Run server
echo "Starting server..."
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

echo "Server is running. Access it at http://localhost:8000"
