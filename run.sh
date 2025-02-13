#!/bin/bash

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Please install Conda first."
    exit 1
fi

# Environment name
ENV_NAME="aipolitician"

# Check if NVIDIA GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "Checking GPU status..."
    nvidia-smi
    HAS_GPU=1
else
    echo "No NVIDIA GPU found. Will proceed with CPU."
    HAS_GPU=0
fi

# Check if environment exists
if ! conda env list | grep -q "^$ENV_NAME "; then
    echo "Creating new conda environment: $ENV_NAME"
    if [ $HAS_GPU -eq 1 ]; then
        # Create environment with GPU support
        conda create -n $ENV_NAME python=3.10 pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
    else
        # Create environment with CPU only
        conda create -n $ENV_NAME python=3.10 pytorch torchvision torchaudio cpuonly -c pytorch -y
    fi
fi

# Activate environment
echo "Activating conda environment: $ENV_NAME"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# Install/upgrade pip requirements
echo "Installing/upgrading pip requirements..."
pip install -r requirements.txt

# Run the FastAPI server
echo "Starting the server..."
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
