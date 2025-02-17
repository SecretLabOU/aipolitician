#!/bin/bash

# Check if conda exists
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Please install Miniconda/Anaconda first."
    exit 1
fi

# Environment setup
ENV_NAME="politician-deep"
REQUIREMENTS="requirements.txt"

# GPU detection
if command -v nvidia-smi &> /dev/null; then
    echo "Detected NVIDIA GPU"
    GPU_ENABLED=1
else
    echo "No NVIDIA GPU detected - using CPU"
    GPU_ENABLED=0
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

# Run server with appropriate settings
if [ $GPU_ENABLED -eq 1 ]; then
    echo "Starting server with GPU support..."
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 \
    --workers 1 \
    --ssl-keyfile=key.pem --ssl-certfile=cert.pem \
    --log-level info
else
    echo "Starting server with CPU only..."
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 \
    --workers 4 \
    --log-level warning
fi