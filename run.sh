#!/bin/bash

# Check if .env exists
if [ ! -f .env ]; then
    echo "Error: .env file not found. Please create one from .env.example"
    exit 1
fi

# Load environment variables first
set -a
source .env
set +a

# Initialize genv
if ! command -v genv &> /dev/null; then
    echo "Error: genv not found. Please install genv first."
    exit 1
fi

# Source genv initialization
if [ -f ~/.genv/env ]; then
    source ~/.genv/env
else
    eval "$(genv shell --init)"
fi

# Check GPU availability
echo "Checking GPU availability..."
GPU_INFO=$(genv devices)
if [ $? -ne 0 ]; then
    echo "Error: Failed to get GPU information"
    exit 1
fi

# Activate specific GPU environment
echo "Setting up GPU environment..."
if ! genv activate --id nat; then
    echo "Error: Failed to activate GPU environment"
    exit 1
fi

# Try to attach to GPU 0
if ! genv attach --index 0; then
    echo "Error: Failed to attach to GPU 0"
    exit 1
fi

# Verify GPU attachment
ATTACHED_GPU=$(genv info | grep "Attached GPU")
if [ -z "$ATTACHED_GPU" ]; then
    echo "Error: No GPU attached after setup"
    exit 1
fi

echo "GPU Setup Complete:"
echo "$(genv info)"

# Environment setup
ENV_NAME="mistral-finetune"

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# Install/update dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Set CUDA device order
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

# Run server with specific worker configuration
echo "Starting server..."
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --workers 1

echo "Server is running. Access it at http://localhost:8000"