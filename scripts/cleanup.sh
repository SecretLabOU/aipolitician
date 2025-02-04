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

# Function to safely remove files/directories
safe_remove() {
    path=$1
    if [ -e "$path" ]; then
        rm -rf "$path"
        if [ $? -eq 0 ]; then
            print_color $GREEN "Removed: $path"
        else
            print_color $RED "Failed to remove: $path"
        fi
    fi
}

# Confirm cleanup
read -p "This will remove all temporary files and reset the application state. Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_color $YELLOW "Cleanup cancelled"
    exit 1
fi

print_color $YELLOW "Starting cleanup..."

# Stop any running containers
if command -v docker >/dev/null 2>&1; then
    print_color $YELLOW "Stopping Docker containers..."
    docker-compose down 2>/dev/null
fi

# Remove database files
print_color $YELLOW "Removing database files..."
safe_remove "data/main.db"
safe_remove "data/cache.db"
safe_remove "data/politicians.db"

# Remove cached data
print_color $YELLOW "Removing cached data..."
safe_remove "data/raw/*"
safe_remove "data/processed/*"
safe_remove "data/embeddings/*"

# Remove downloaded models (optional)
read -p "Remove downloaded models? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_color $YELLOW "Removing downloaded models..."
    safe_remove "models/*"
fi

# Remove Python cache
print_color $YELLOW "Removing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete
find . -type f -name "*.pyd" -delete

# Remove test cache
print_color $YELLOW "Removing test cache..."
safe_remove ".pytest_cache"
safe_remove ".coverage"
safe_remove "htmlcov"

# Remove logs
print_color $YELLOW "Removing logs..."
safe_remove "*.log"
safe_remove "logs/*"

# Remove monitoring data
print_color $YELLOW "Removing monitoring data..."
safe_remove "monitoring/grafana/data"

# Reset environment
if [ -f ".env" ]; then
    print_color $YELLOW "Backing up .env file to .env.backup..."
    cp .env .env.backup
    rm .env
    print_color $GREEN "Environment file backed up and removed"
fi

print_color $GREEN "\nCleanup complete!"
print_color $YELLOW "To rebuild the application:"
echo "1. Run ./setup.sh to set up the environment"
echo "2. Edit .env with your configuration"
echo "3. Run python scripts/setup_models.py to download models"
echo "4. Run python scripts/collect_data.py to initialize data"
