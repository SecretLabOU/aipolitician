#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Print with color
print_color() {
    color=$1
    message=$2
    printf "${color}%s${NC}\n" "$message"
}

# Initialize conda environment
init_conda() {
    local env_name=$1
    
    print_color $YELLOW "Initializing conda environment: $env_name"
    
    # Source conda.sh
    CONDA_BASE=$(dirname $(dirname $(which python)))
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    
    if [ $? -ne 0 ]; then
        # Try alternative conda paths
        if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
            source "$HOME/miniconda3/etc/profile.d/conda.sh"
        elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
            source "$HOME/anaconda3/etc/profile.d/conda.sh"
        else
            print_color $RED "Could not find conda.sh. Please ensure conda is installed."
            exit 1
        fi
    fi
    
    # Activate environment
    conda activate "$env_name"
    if [ $? -ne 0 ]; then
        print_color $RED "Failed to activate conda environment"
        exit 1
    fi
    
    print_color $GREEN "Conda environment ready"
    
    # Verify Python environment
    python --version
    which python
}

# Initialize database
init_database() {
    print_color $YELLOW "Initializing database..."
    cd "${PROJECT_ROOT}"
    
    # Set PYTHONPATH
    export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
    
    # Get database URL from .env file
    if [ ! -f ".env" ]; then
        print_color $RED ".env file not found"
        exit 1
    fi
    
    # Extract database connection details from DATABASE_URL
    DB_URL=$(grep -oP 'DATABASE_URL=\K[^#\s]+' .env)
    if [ -z "$DB_URL" ]; then
        print_color $RED "DATABASE_URL not found in .env file"
        exit 1
    fi
    
    # Parse database URL to get credentials
    DB_USER=$(echo $DB_URL | sed -n 's/.*:\/\/\([^:]*\):.*/\1/p')
    DB_PASS=$(echo $DB_URL | sed -n 's/.*:\/\/[^:]*:\([^@]*\)@.*/\1/p')
    DB_HOST=$(echo $DB_URL | sed -n 's/.*@\([^:]*\):.*/\1/p')
    DB_PORT=$(echo $DB_URL | sed -n 's/.*:\([^/]*\)\/.*/\1/p')
    DB_NAME=$(echo $DB_URL | sed -n 's/.*\/\(.*\)/\1/p')
    
    # Create migrations versions directory if it doesn't exist
    mkdir -p migrations/versions
    
    # Remove existing migrations
    print_color $YELLOW "Cleaning up old migrations..."
    rm -rf migrations/versions/*

    # Create tables directly
    print_color $YELLOW "Creating tables..."
    PYTHONPATH="${PROJECT_ROOT}" python << EOF
from src.database.models import Base
from sqlalchemy import create_engine
from src.config import get_database_url

engine = create_engine(get_database_url())
Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)
EOF
    
    if [ $? -eq 0 ]; then
        print_color $GREEN "Database initialized successfully"
    else
        print_color $RED "Error initializing database"
        exit 1
    fi
}

# Show database status
show_database_status() {
    print_color $YELLOW "\nDatabase Status:"
    print_color $YELLOW "----------------"
    
    # Get database URL from .env file
    DB_URL=$(grep -oP 'DATABASE_URL=\K[^#\s]+' .env)
    
    # Parse database URL to get credentials
    DB_USER=$(echo $DB_URL | sed -n 's/.*:\/\/\([^:]*\):.*/\1/p')
    DB_PASS=$(echo $DB_URL | sed -n 's/.*:\/\/[^:]*:\([^@]*\)@.*/\1/p')
    DB_HOST=$(echo $DB_URL | sed -n 's/.*@\([^:]*\):.*/\1/p')
    DB_PORT=$(echo $DB_URL | sed -n 's/.*:\([^/]*\)\/.*/\1/p')
    DB_NAME=$(echo $DB_URL | sed -n 's/.*\/\(.*\)/\1/p')
    
    # Connect to database and show statistics
    PGPASSWORD=$DB_PASS psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME << EOF
\echo '\nTable Statistics:'
SELECT schemaname, relname, n_live_tup 
FROM pg_stat_user_tables 
ORDER BY n_live_tup DESC;

\echo '\nChat History:'
SELECT session_id, COUNT(*) as message_count, 
       MIN(created_at) as first_message,
       MAX(created_at) as last_message
FROM chat_history 
GROUP BY session_id 
ORDER BY last_message DESC;
EOF
}

# Main function
main() {
    if [ "$#" -ne 1 ]; then
        print_color $RED "Usage: $0 <conda-env-name>"
        exit 1
    fi
    
    local conda_env=$1
    
    # Verify .env file exists
    if [ ! -f ".env" ]; then
        print_color $RED ".env file not found. Please create it from .env.example"
        exit 1
    fi
    
    # Initialize conda environment
    init_conda "$conda_env"
    
    # Initialize database
    init_database
    
    # Show database status
    show_database_status
    
    print_color $GREEN "\nDatabase initialization complete!"
    print_color $YELLOW "\nTo inspect the database manually, use the connection details from your .env file"
}

# Show help
show_help() {
    echo "Usage: $0 <conda-env-name>"
    echo
    echo "This script initializes the PoliticianAI database."
    echo
    echo "Arguments:"
    echo "  conda-env-name: Name of the conda environment to use"
    echo
    echo "Before running:"
    echo "1. Ensure you are on the GPU server"
    echo "2. Verify .env file is properly configured"
    echo "3. Make sure PostgreSQL is running on port 35432"
    echo
    echo "Example:"
    echo "  ./init_database.sh poly-2"
}

# Check if help is requested
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    show_help
    exit 0
fi

# Run main function
main "$@"
