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

# Check if session name is provided
if [ "$#" -ne 1 ]; then
    print_color $RED "Usage: $0 <session-name>"
    exit 1
fi

SESSION_NAME=$1

# Check if genv is available
if ! command -v genv >/dev/null 2>&1; then
    print_color $RED "genv command not found. Please ensure it's installed."
    exit 1
fi

# Initialize genv shell
print_color $YELLOW "Initializing genv shell..."
eval "$(genv shell --init)"

# Check if session exists
if ! genv list | grep -q "$SESSION_NAME"; then
    print_color $RED "Session '$SESSION_NAME' not found"
    exit 1
fi

# Deactivate session
print_color $YELLOW "Deactivating GPU session '$SESSION_NAME'..."
genv deactivate --id "$SESSION_NAME"

if [ $? -eq 0 ]; then
    print_color $GREEN "Successfully deactivated session '$SESSION_NAME'"
else
    print_color $RED "Failed to deactivate session '$SESSION_NAME'"
    exit 1
fi

# Clean up temporary files
print_color $YELLOW "Cleaning up temporary files..."

# Clean Python cache files
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type f -name "*.pyo" -delete 2>/dev/null
find . -type f -name "*.pyd" -delete 2>/dev/null

# Clean test cache
rm -rf .pytest_cache/ 2>/dev/null
rm -rf .coverage 2>/dev/null
rm -rf htmlcov/ 2>/dev/null

# Clean logs
rm -f app.log 2>/dev/null
rm -f *.log 2>/dev/null

# Clean temporary model files
find models/ -type f -name "*.tmp" -delete 2>/dev/null

print_color $GREEN "Cleanup complete!"

# Show current GPU status
print_color $YELLOW "\nCurrent GPU status:"
nvidia-smi

print_color $YELLOW "\nCurrent GPU sessions:"
genv devices
