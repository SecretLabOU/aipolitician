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

# Initialize genv shell first
init_genv

# Show current GPU status
print_color $YELLOW "Current GPU Status:"
nvidia-smi

print_color $YELLOW "\nCurrent GPU Sessions:"
genv devices

# Ask for session name
read -p "Enter the session name to deactivate (or 'all' for all sessions): " session_name

if [ "$session_name" = "all" ]; then
    print_color $YELLOW "Deactivating all GPU sessions..."
    
    # Get list of active sessions
    active_sessions=$(genv devices | grep -oE '[^ ]+$' | sort -u)
    
    if [ -z "$active_sessions" ]; then
        print_color $YELLOW "No active sessions found"
    else
        # Deactivate each session
        for session in $active_sessions; do
            if [ ! -z "$session" ]; then
                print_color $YELLOW "Deactivating session: $session"
                genv deactivate --id "$session"
                
                if [ $? -eq 0 ]; then
                    print_color $GREEN "Session $session deactivated successfully"
                else
                    print_color $RED "Error deactivating session: $session"
                fi
            fi
        done
    fi
    
    print_color $GREEN "All sessions processed"
else
    print_color $YELLOW "Deactivating session: $session_name"
    genv deactivate --id "$session_name"
    
    if [ $? -eq 0 ]; then
        print_color $GREEN "Session deactivated successfully"
    else
        print_color $RED "Error deactivating session"
        exit 1
    fi
fi

# Show final status
print_color $YELLOW "\nFinal GPU Status:"
nvidia-smi

print_color $YELLOW "\nRemaining GPU Sessions:"
genv devices

print_color $GREEN "\nCleanup complete!"
print_color $YELLOW "Note: Don't forget to:"
echo "1. Deactivate conda environment: conda deactivate"
echo "2. Add to your ~/.bashrc if you haven't: eval \"\$(genv shell --init)\""
