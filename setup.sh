#!/bin/bash

# Exit on error
set -e

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

# Check if running with sudo
if [ "$EUID" -eq 0 ]; then
    print_color $RED "Please do not run this script with sudo"
    exit 1
fi

# Check Python version
python3 --version >/dev/null 2>&1 || {
    print_color $RED "Python 3 is required but not installed. Aborting."
    exit 1
}

# Check if CUDA is available
nvidia-smi >/dev/null 2>&1
CUDA_AVAILABLE=$?

if [ $CUDA_AVAILABLE -eq 0 ]; then
    print_color $GREEN "CUDA is available"
else
    print_color $YELLOW "CUDA is not available. Will use CPU mode (slower)"
fi

# Create virtual environment
print_color $GREEN "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
print_color $GREEN "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
print_color $GREEN "Installing dependencies..."
pip install -r requirements.txt

# Install development dependencies if --dev flag is provided
if [[ "$*" == *"--dev"* ]]; then
    print_color $GREEN "Installing development dependencies..."
    pip install -r requirements-dev.txt
fi

# Create necessary directories
print_color $GREEN "Creating project directories..."
mkdir -p data/{raw,processed,embeddings} models

# Initialize database
print_color $GREEN "Initializing database..."
python3 scripts/collect_data.py

# Download models
print_color $GREEN "Downloading required models..."
python3 scripts/setup_models.py

# Set up pre-commit hooks if in dev mode
if [[ "$*" == *"--dev"* ]]; then
    print_color $GREEN "Setting up pre-commit hooks..."
    pre-commit install
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    print_color $GREEN "Creating .env file..."
    cp .env.example .env
    print_color $YELLOW "Please edit .env file with your configuration"
fi

# Check if Docker is installed
if command -v docker >/dev/null 2>&1; then
    print_color $GREEN "Docker is available"
    
    # Check if user wants to build Docker image
    if [[ "$*" == *"--docker"* ]]; then
        print_color $GREEN "Building Docker image..."
        docker-compose build
        
        if [[ "$*" == *"--dev"* ]]; then
            print_color $GREEN "Starting development services..."
            docker-compose --profile dev up -d
        else
            print_color $GREEN "Starting basic services..."
            docker-compose up -d
        fi
    fi
else
    print_color $YELLOW "Docker is not installed. Skipping Docker setup."
fi

print_color $GREEN "Setup complete!"
print_color $GREEN "To activate the virtual environment, run: source venv/bin/activate"
print_color $GREEN "To start the application, run: python main.py"

if [ ! -f models/llama-2-7b-chat.gguf ]; then
    print_color $YELLOW "Note: You need to manually download the LLaMA 2 model."
    print_color $YELLOW "Please visit: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF"
    print_color $YELLOW "Download the model and place it in the models directory as llama-2-7b-chat.gguf"
fi

# Print help information
print_color $GREEN "\nAvailable commands:"
echo "python main.py            - Start the application"
echo "pytest tests/            - Run tests"
echo "docker-compose up        - Start with Docker"
echo "docker-compose down      - Stop Docker containers"

if [[ "$*" == *"--dev"* ]]; then
    echo "black .                 - Format code"
    echo "isort .                - Sort imports"
    echo "mypy .                 - Type checking"
    echo "pytest --cov=src tests/ - Run tests with coverage"
fi
