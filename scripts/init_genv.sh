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

# Check if genv command exists
if ! command -v genv >/dev/null 2>&1; then
    print_color $RED "genv command not found. Please ensure it's installed."
    exit 1
fi

# Check if initialization line already exists in .bashrc
if grep -q "eval \"\$(genv shell --init)\"" ~/.bashrc; then
    print_color $YELLOW "genv shell initialization already exists in ~/.bashrc"
else
    print_color $YELLOW "Adding genv shell initialization to ~/.bashrc..."
    echo 'eval "$(genv shell --init)"' >> ~/.bashrc
    print_color $GREEN "Added successfully"
fi

# Initialize current shell
print_color $YELLOW "Initializing current shell..."
eval "$(genv shell --init)"

if [ $? -eq 0 ]; then
    print_color $GREEN "genv shell initialized successfully"
    print_color $YELLOW "\nYou can now run:"
    echo "./scripts/run_on_gpu.sh <your-session-name>"
else
    print_color $RED "Failed to initialize genv shell"
    exit 1
fi
