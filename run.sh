#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print with color
print_color() {
    color=$1
    message=$2
    printf "${color}%s${NC}\n" "$message"
}

# Initialize genv shell
init_genv() {
    print_color $YELLOW "Initializing genv shell environment..."
    
    if ! command -v genv >/dev/null 2>&1; then
        print_color $RED "genv command not found. Please ensure it's installed."
        exit 1
    fi
    
    eval "$(genv shell --init)"
    
    if [ $? -eq 0 ]; then
        print_color $GREEN "genv shell initialized successfully"
    else
        print_color $RED "Failed to initialize genv shell"
        exit 1
    fi
}

# Check GPU availability
check_gpu() {
    print_color $YELLOW "Checking GPU status..."
    nvidia-smi
    
    print_color $YELLOW "\nChecking current GPU sessions..."
    genv devices
    
    if ! nvidia-smi &>/dev/null; then
        print_color $RED "No GPUs found or nvidia-smi not available"
        exit 1
    fi
}

# Setup GPU environment
setup_gpu() {    
    print_color $YELLOW "\nDeactivating any existing sessions..."
    genv deactivate --all 2>/dev/null
    
    print_color $YELLOW "\nActivating GPU session..."
    genv activate --id nat
    
    if [ $? -ne 0 ]; then
        print_color $RED "Failed to activate GPU session"
        exit 1
    fi
    
    print_color $YELLOW "\nAttaching GPU..."
    genv attach --count 1
    
    if [ $? -ne 0 ]; then
        print_color $RED "Failed to attach GPU"
        genv deactivate --id nat
        exit 1
    fi
    
    print_color $GREEN "GPU session setup complete"
    
    # Show current GPU status
    print_color $YELLOW "\nCurrent GPU status:"
    genv devices
}

# Main setup
main() {
    # Check if .env exists
    if [ ! -f .env ]; then
        print_color $RED "Error: .env file not found. Please create one from .env.example"
        exit 1
    fi
    
    # Load environment variables
    set -a
    source .env
    set +a
    
    # Initialize genv
    init_genv
    
    # Check GPU status
    check_gpu
    
    # Setup GPU environment
    setup_gpu
    
    # Activate conda environment
    print_color $YELLOW "Activating conda environment: mistral-finetune"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate mistral-finetune
    
    if [ $? -ne 0 ]; then
        print_color $RED "Failed to activate conda environment: mistral-finetune"
        exit 1
    fi
    
    # Set CUDA device order
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES=0
    
    # Start server
    print_color $GREEN "Starting server..."
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --workers 1
}

# Run main function
main