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
    
    # Deactivate each session
    for session in $active_sessions; do
        print_color $YELLOW "Deactivating session: $session"
        genv deactivate --id "$session"
    done
    
    print_color $GREEN "All sessions deactivated"
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
print_color $YELLOW "Note: Don't forget to deactivate your virtual environment:"
echo "deactivate"
