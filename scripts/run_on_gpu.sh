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
    
    # Check if genv shell is initialized
    if ! command -v genv >/dev/null 2>&1; then
        print_color $RED "genv command not found. Please ensure it's installed."
        exit 1
    fi
    
    # Initialize genv shell
    eval "$(genv shell --init)"
    
    if [ $? -eq 0 ]; then
        print_color $GREEN "genv shell initialized successfully"
    else
        print_color $RED "Failed to initialize genv shell"
        exit 1
    fi
}

# Check conda environment
check_conda() {
    local env_name=$1
    
    # Check if conda is available
    if ! command -v conda >/dev/null 2>&1; then
        print_color $RED "conda not found. Please install miniconda or anaconda."
        exit 1
    fi
    
    # Initialize conda for shell
    print_color $YELLOW "Initializing conda..."
    eval "$(conda shell.bash hook)"
    
    # Check if environment exists
    if conda env list | grep -q "^${env_name}[[:space:]]"; then
        print_color $GREEN "Found existing conda environment: $env_name"
    else
        print_color $YELLOW "Creating new conda environment: $env_name"
        conda create -y -n "$env_name" python=3.8
        if [ $? -ne 0 ]; then
            print_color $RED "Failed to create conda environment"
            exit 1
        fi
    fi
    
    # Activate environment
    print_color $YELLOW "Activating conda environment: $env_name"
    conda activate "$env_name"
    if [ $? -ne 0 ]; then
        print_color $RED "Failed to activate conda environment"
        exit 1
    fi
    
    print_color $GREEN "Conda environment ready"
}

# Check GPU availability
check_gpu() {
    print_color $YELLOW "Checking GPU status..."
    nvidia-smi
    
    print_color $YELLOW "\nChecking current GPU sessions..."
    genv devices
    
    # Verify GPUs are available
    if ! nvidia-smi &>/dev/null; then
        print_color $RED "No GPUs found or nvidia-smi not available"
        exit 1
    fi
}

# Setup GPU environment
setup_gpu() {
    local session_name=$1
    
    print_color $YELLOW "\nActivating GPU session..."
    genv activate --id "$session_name"
    
    if [ $? -ne 0 ]; then
        print_color $RED "Failed to activate GPU session"
        exit 1
    fi
    
    print_color $YELLOW "\nAttaching GPU..."
    genv attach --count 1
    
    if [ $? -ne 0 ]; then
        print_color $RED "Failed to attach GPU"
        genv deactivate --id "$session_name"
        exit 1
    fi
    
    print_color $GREEN "GPU session setup complete"
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
    python scripts/setup_models.py
    
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
    python scripts/collect_data.py
    
    if [ $? -eq 0 ]; then
        print_color $GREEN "Data initialized successfully"
    else
        print_color $RED "Error initializing data"
        exit 1
    fi
}

# Main setup function
main() {
    if [ "$#" -ne 2 ]; then
        print_color $RED "Usage: $0 <conda-env-name> <session-name>"
        exit 1
    fi
    
    local conda_env=$1
    local session_name=$2
    
    # Print system info
    print_color $YELLOW "System Information:"
    python --version
    conda --version
    
    # Initialize genv shell first
    init_genv
    
    # Check GPU status
    check_gpu
    
    # Setup and activate conda environment
    check_conda "$conda_env"
    
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
    python main.py
}

# Show help
show_help() {
    echo "Usage: $0 <conda-env-name> <session-name>"
    echo
    echo "This script sets up and runs PoliticianAI on a GPU server."
    echo
    echo "Arguments:"
    echo "  conda-env-name: Name of the conda environment to use/create"
    echo "  session-name:   Name for the GPU session"
    echo
    echo "Before running:"
    echo "1. SSH into the GPU server"
    echo "2. Clone the repository"
    echo "3. Navigate to the project directory"
    echo "4. Ensure miniconda/anaconda is installed"
    echo "5. Run this script with environment and session names"
    echo
    echo "Example:"
    echo "  ./run_on_gpu.sh politician-ai my-session"
    echo
    echo "The script will:"
    echo "- Initialize genv shell environment"
    echo "- Check GPU availability"
    echo "- Set up/activate conda environment"
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
