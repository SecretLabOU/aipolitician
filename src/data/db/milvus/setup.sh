#!/bin/bash

# Setup script for Milvus database
# This script initializes the Milvus database for the AI Politician project

# Exit on error
set -e

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print section header
section() {
    echo -e "\n${GREEN}=== $1 ===${NC}\n"
}

# Print information messages
info() {
    echo -e "${YELLOW}INFO:${NC} $1"
}

# Print error messages
error() {
    echo -e "${RED}ERROR:${NC} $1"
}

# Get the user's home directory
USER_HOME="$HOME"
DB_DIR="$USER_HOME/Databases/ai_politician_milvus"

# Check if Docker is installed
section "Checking Prerequisites"
if ! command -v docker &> /dev/null; then
    error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check for docker compose (try both formats)
if ! command -v docker-compose &> /dev/null && ! command -v docker compose &> /dev/null; then
    error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create network if it doesn't exist
section "Creating Docker Network"
if ! docker network ls | grep -q milvus; then
    info "Creating 'milvus' Docker network"
    docker network create milvus
else
    info "'milvus' Docker network already exists"
fi

# Create database directories
section "Creating Database Directories"
info "Setting up directories at $DB_DIR"

mkdir -p "$DB_DIR/data"
mkdir -p "$DB_DIR/etcd"
mkdir -p "$DB_DIR/minio"
mkdir -p "$DB_DIR/logs"

# Set proper permissions
info "Setting directory permissions"
chmod -R 755 "$DB_DIR"

# Update docker-compose.yml file with correct paths
section "Updating Docker Compose File"
info "Updating paths in docker-compose.yml"

# Use sed to replace paths in the docker-compose.yml file
sed -i.bak "s|/home/natalie/Databases/ai_politician_milvus|$DB_DIR|g" docker-compose.yml
info "Docker Compose file updated with your home directory"

# Start Milvus with Docker Compose
section "Starting Milvus Database"
info "Starting Milvus container using Docker Compose"
cd "$(dirname "$0")"

# Try docker compose first, then docker-compose if that fails
if command -v docker compose &> /dev/null; then
    docker compose up -d
elif command -v docker-compose &> /dev/null; then
    docker-compose up -d
else
    error "Neither docker compose nor docker-compose commands are available"
    exit 1
fi

# Wait for Milvus to be ready
info "Waiting for Milvus to be ready..."
sleep 10

# Check if Milvus is running
if docker ps | grep -q ai_politician_milvus; then
    info "Milvus container is running"
else
    error "Milvus container is not running. Please check docker logs."
    exit 1
fi

# Success message
section "Setup Complete"
echo -e "${GREEN}Milvus database setup is complete!${NC}"
echo -e "Milvus server is running at: localhost:19530"
echo -e "You can now use the Python scripts to create collections and indexes."
echo -e "\nTo stop Milvus: docker compose down (or docker-compose down)"
echo -e "To view logs: docker logs ai_politician_milvus"
