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

# Check if genv is installed
if command -v genv >/dev/null 2>&1; then
    print_color $GREEN "genv is already installed"
else
    print_color $YELLOW "Installing genv..."
    
    # Install genv
    pip install genv
    
    if [ $? -eq 0 ]; then
        print_color $GREEN "genv installed successfully"
    else
        print_color $RED "Failed to install genv"
        exit 1
    fi
fi

# Initialize genv shell
print_color $YELLOW "Initializing genv shell..."

# Add genv initialization to shell config
SHELL_CONFIG="$HOME/.bashrc"
if [ -f "$HOME/.zshrc" ]; then
    SHELL_CONFIG="$HOME/.zshrc"
fi

# Check if genv init is already in shell config
if grep -q "genv shell --init" "$SHELL_CONFIG"; then
    print_color $GREEN "genv shell already initialized in $SHELL_CONFIG"
else
    # Add genv initialization
    echo '# Initialize genv shell' >> "$SHELL_CONFIG"
    echo 'eval "$(genv shell --init)"' >> "$SHELL_CONFIG"
    print_color $GREEN "Added genv initialization to $SHELL_CONFIG"
fi

# Initialize genv shell in current session
eval "$(genv shell --init)"

print_color $GREEN "genv shell initialized successfully"
print_color $YELLOW "Please run 'source $SHELL_CONFIG' or start a new shell session"

# Show current GPU status
print_color $YELLOW "\nCurrent GPU status:"
nvidia-smi

print_color $YELLOW "\nCurrent GPU sessions:"
genv devices
