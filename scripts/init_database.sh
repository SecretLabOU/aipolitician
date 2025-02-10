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
    
    # Drop database if exists and create new one
    print_color $YELLOW "Setting up database..."
    PGPASSWORD=preston psql -h localhost -p 35432 -U nat -d postgres << EOF
DROP DATABASE IF EXISTS politician_ai;
CREATE DATABASE politician_ai;
EOF
    
    if [ $? -ne 0 ]; then
        print_color $RED "Error creating database"
        exit 1
    fi
    
    # Wait a moment for the database to be ready
    sleep 2
    
    # Create migrations versions directory if it doesn't exist
    mkdir -p migrations/versions
    
    # Generate initial migration
    print_color $YELLOW "Generating initial migration..."
    PYTHONPATH="${PROJECT_ROOT}" alembic revision --autogenerate -m "Initial migration"
    
    # Apply migration
    print_color $YELLOW "Applying migration..."
    PYTHONPATH="${PROJECT_ROOT}" alembic upgrade head
    
    # Run data collection script
    print_color $YELLOW "Collecting politician data..."
    PYTHONPATH="${PROJECT_ROOT}" python scripts/collect_politician_data.py
    
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
    
    # Connect to database and show statistics
    PGPASSWORD=preston psql -h localhost -p 35432 -U nat -d politician_ai << EOF
\echo '\nTable Statistics:'
SELECT schemaname, relname, n_live_tup 
FROM pg_stat_user_tables 
ORDER BY n_live_tup DESC;

\echo '\nPoliticians:'
SELECT name, party, position FROM politicians;

\echo '\nTopics:'
SELECT name, (
    SELECT COUNT(*) 
    FROM statements 
    WHERE topic_id = topics.id
) as statement_count 
FROM topics 
ORDER BY statement_count DESC;

\echo '\nRecent Statements:'
SELECT p.name, t.name as topic, s.content, s.sentiment_score, s.date
FROM statements s
JOIN politicians p ON s.politician_id = p.id
JOIN topics t ON s.topic_id = t.id
ORDER BY s.date DESC
LIMIT 5;
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
    print_color $YELLOW "\nTo inspect the database manually, use:"
    print_color $NC "psql -h localhost -p 35432 -U nat -d politician_ai"
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
