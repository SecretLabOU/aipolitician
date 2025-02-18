#!/bin/bash

# Check if .env exists
if [ ! -f .env ]; then
    echo "Error: .env file not found. Please create one from .env.example"
    exit 1
fi

# Environment setup
ENV_NAME="politician-deep"
REQUIREMENTS="requirements.txt"

# GPU detection and setup
if command -v nvidia-smi &> /dev/null; then
    echo "Detected NVIDIA GPU"
    GPU_ENABLED=1
    
    # Get GPU memory (in MiB)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
    
    # Set number of workers based on GPU memory
    # Large models need dedicated GPU memory
    WORKERS=1
    
    # Check if we're on the shared GPU server
    if [ -d "/home/shared_models/aipolitician" ]; then
        echo "Detected shared GPU environment"
        # Ensure model path is accessible
        if [ ! -d "/home/shared_models/aipolitician/fine_tuned_trump_mistral" ]; then
            echo "Error: Trump model not found in shared models directory"
            exit 1
        fi
    fi
else
    echo "No NVIDIA GPU detected - using CPU"
    GPU_ENABLED=0
    WORKERS=4
fi

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

# Run server with appropriate settings
if [ $GPU_ENABLED -eq 1 ]; then
    echo "Starting server with GPU support..."
    # Use --timeout 300 for longer model loading time
    # Use --limit-concurrency for memory management
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 \
    --workers $WORKERS \
    --timeout 300 \
    --limit-concurrency 5 \
    --log-level info
else
    echo "Starting server with CPU only..."
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 \
    --workers $WORKERS \
    --log-level warning
fi

echo "Server is running. Access it at http://localhost:8000"
