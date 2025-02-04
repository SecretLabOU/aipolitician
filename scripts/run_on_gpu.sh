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

# Check if virtual environment exists
check_venv() {
    if [ ! -d "venv" ]; then
        print_color $YELLOW "Creating virtual environment..."
        python3 -m venv venv
    fi
}

# Check GPU availability
check_gpu() {
    print_color $YELLOW "Checking GPU status..."
    nvidia-smi
    
    print_color $YELLOW "\nChecking current GPU sessions..."
    genv devices
}

# Setup GPU environment
setup_gpu() {
    local session_name=$1
    
    print_color $YELLOW "\nActivating GPU session..."
    genv activate --id "$session_name"
    
    print_color $YELLOW "\nAttaching GPU..."
    genv attach --count 1
}

# Install dependencies
install_deps() {
    print_color $YELLOW "Installing dependencies..."
    pip install -r requirements.txt
    
    if [ $? -eq 0 ]; then
        print_color $GREEN "Dependencies installed successfully"
    else
        print_color $RED "Error installing dependencies"
        exit 1
    fi
}

# Download and set up models
setup_models() {
    print_color $YELLOW "Setting up models..."
    python3 scripts/setup_models.py
    
    if [ $? -eq 0 ]; then
        print_color $GREEN "Models set up successfully"
    else
        print_color $RED "Error setting up models"
        exit 1
    fi
}

# Initialize data
init_data() {
    print_color $YELLOW "Initializing data..."
    python3 scripts/collect_data.py
    
    if [ $? -eq 0 ]; then
        print_color $GREEN "Data initialized successfully"
    else
        print_color $RED "Error initializing data"
        exit 1
    fi
}

# Main setup function
main() {
    if [ "$#" -ne 1 ]; then
        print_color $RED "Usage: $0 <session-name>"
        exit 1
    fi
    
    local session_name=$1
    
    # Print system info
    print_color $YELLOW "System Information:"
    python3 --version
    pip --version
    
    # Check GPU status
    check_gpu
    
    # Setup virtual environment
    check_venv
    source venv/bin/activate
    
    # Setup GPU environment
    setup_gpu "$session_name"
    
    # Install dependencies
    install_deps
    
    # Create .env if it doesn't exist
    if [ ! -f ".env" ]; then
        print_color $YELLOW "Creating .env file..."
        cp .env.example .env
        print_color $GREEN "Created .env file. Please edit it with your configuration."
        exit 1
    fi
    
    # Setup models and data
    setup_models
    init_data
    
    # Start the application
    print_color $GREEN "\nSetup complete! Starting application..."
    python3 main.py
}

# Show help
show_help() {
    echo "Usage: $0 <session-name>"
    echo
    echo "This script sets up and runs PoliticianAI on a GPU server."
    echo
    echo "Before running:"
    echo "1. SSH into the GPU server"
    echo "2. Clone the repository"
    echo "3. Navigate to the project directory"
    echo "4. Run this script with your session name"
    echo
    echo "Example:"
    echo "  ./run_on_gpu.sh my-session"
    echo
    echo "The script will:"
    echo "- Check GPU availability"
    echo "- Set up virtual environment"
    echo "- Install dependencies"
    echo "- Download required models"
    echo "- Initialize data"
    echo "- Start the application"
}

# Check if help is requested
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    show_help
    exit 0
fi

# Run main function
main "$@"
