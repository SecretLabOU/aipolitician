#!/bin/bash

# Check if .env exists
if [ ! -f .env ]; then
    echo "Error: .env file not found. Please create one from .env.example"
    exit 1
fi

# Initialize genv shell if not already done
if ! command -v genv &> /dev/null; then
    echo "Initializing genv shell..."
    eval "$(genv shell --init)"
fi

# Environment setup
ENV_NAME="mistral-finetune"

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# Install/update dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Load environment variables
set -a
source .env
set +a

# Setup GPU environment with error handling
echo "Setting up GPU..."
if ! genv activate --id nat 2>/dev/null; then
    echo "Warning: Could not activate GPU environment. Continuing without specific GPU allocation."
fi

if ! genv attach --index 0 2>/dev/null; then
    echo "Warning: Could not attach to GPU 0. Continuing with default GPU assignment."
fi

# Set CUDA device order to prevent NVML issues
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Run server with specific worker configuration
echo "Starting server..."
CUDA_VISIBLE_DEVICES=0 uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --workers 1

echo "Server is running. Access it at http://localhost:8000"