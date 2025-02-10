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
        conda create -y -n "$env_name" python=3.10.6 pip
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

# Install dependencies
install_deps() {
    print_color $YELLOW "Installing dependencies..."
    
    # Upgrade pip first
    print_color $YELLOW "Upgrading pip..."
    pip install --upgrade pip
    
    # Install dependencies
    print_color $YELLOW "Installing project dependencies..."
    if pip install -r requirements.txt; then
        print_color $GREEN "Dependencies installed successfully"
    else
        print_color $RED "Error installing dependencies"
        exit 1
    fi
    
    # Install project in development mode
    print_color $YELLOW "Installing project in development mode..."
    cd "${PROJECT_ROOT}"
    pip install -e .
    if [ $? -ne 0 ]; then
        print_color $RED "Failed to install project in development mode"
        exit 1
    fi
    print_color $GREEN "Project installed in development mode"
}

# Initialize database
init_database() {
    print_color $YELLOW "Initializing database..."
    cd "${PROJECT_ROOT}"
    
    # Create migrations versions directory if it doesn't exist
    mkdir -p migrations/versions
    
    # Generate initial migration
    print_color $YELLOW "Generating initial migration..."
    alembic revision --autogenerate -m "Initial migration"
    
    # Apply migration
    print_color $YELLOW "Applying migration..."
    alembic upgrade head
    
    # Run data collection script
    print_color $YELLOW "Collecting politician data..."
    python scripts/collect_politician_data.py
    
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
    PGPASSWORD=$DB_PASSWORD psql -h localhost -p 35432 -U politician_ai_user -d politician_ai << EOF
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

# Main setup function
main() {
    if [ "$#" -ne 2 ]; then
        print_color $RED "Usage: $0 <conda-env-name> <session-name>"
        exit 1
    fi
    
    local conda_env=$1
    local session_name=$2
    
    # Initialize genv shell
    init_genv
    
    # Setup and activate conda environment
    check_conda "$conda_env"
    
    # Install dependencies
    install_deps
    
    # Initialize database
    init_database
    
    # Show database status
    show_database_status
    
    print_color $GREEN "\nDatabase initialization complete!"
    print_color $YELLOW "\nTo inspect the database manually, use:"
    print_color $NC "psql -h localhost -p 35432 -U politician_ai_user -d politician_ai"
}

# Show help
show_help() {
    echo "Usage: $0 <conda-env-name> <session-name>"
    echo
    echo "This script initializes the PoliticianAI database on the GPU server."
    echo
    echo "Arguments:"
    echo "  conda-env-name: Name of the conda environment to use/create"
    echo "  session-name:   Name for the GPU session"
    echo
    echo "Before running:"
    echo "1. Ensure you are on the GPU server"
    echo "2. Verify .env file is properly configured"
    echo "3. Make sure PostgreSQL is running on port 35432"
    echo
    echo "Example:"
    echo "  ./init_database.sh politician-ai my-session"
}

# Check if help is requested
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    show_help
    exit 0
fi

# Run main function
main "$@"
