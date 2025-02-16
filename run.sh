#!/bin/bash

# Function to check Python version
check_python_version() {
    if command -v python3.10 &> /dev/null; then
        return 0
    elif command -v python3 &> /dev/null; then
        python3 -c 'import sys; exit(0 if sys.version_info >= (3, 10) else 1)' &> /dev/null
        return $?
    else
        return 1
    fi
}

# Function to create virtual environment
create_venv() {
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
}

# Function to setup GPU environment
setup_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        echo "Checking GPU status..."
        nvidia-smi
        
        if command -v genv &> /dev/null; then
            echo "Setting up genv..."
            genv activate --id aipolitician
            genv attach --count 1
            export CUDA_VISIBLE_DEVICES=0
            export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
            HAS_GPU=1
        else
            echo "genv not found, but GPU is available. Will use GPU directly."
            HAS_GPU=1
        fi
    else
        echo "No NVIDIA GPU found. Will proceed with CPU."
        HAS_GPU=0
    fi
}

# Main execution starts here
echo "Setting up AI Politician environment..."

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "Using Conda for environment management..."
    
    # Environment name
    ENV_NAME="aipolitician"
    
    # Create/update conda environment
    if ! conda env list | grep -q "^$ENV_NAME "; then
        echo "Creating new conda environment: $ENV_NAME"
        conda create -n $ENV_NAME python=3.10 -y
    fi
    
    # Activate environment
    echo "Activating conda environment: $ENV_NAME"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate $ENV_NAME
    
else
    echo "Conda not found. Using virtual environment instead..."
    
    # Check Python version
    if ! check_python_version; then
        echo "Error: Python 3.10 or higher is required."
        exit 1
    fi
    
    # Create and activate virtual environment if it doesn't exist
    if [ ! -d ".venv" ]; then
        create_venv
    else
        source .venv/bin/activate
    fi
fi

# Setup GPU environment
setup_gpu

# Install/upgrade pip requirements
echo "Installing/upgrading pip requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# Create cache directories
mkdir -p ~/.cache/aipolitician/models
mkdir -p ~/.cache/aipolitician/sessions

# Run the FastAPI server
echo "Starting the server..."
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
