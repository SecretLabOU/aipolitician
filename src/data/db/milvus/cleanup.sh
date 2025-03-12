#!/bin/bash

# Cleanup script for Milvus database
# This script stops Docker containers and optionally removes data

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

# Function to ask for confirmation
confirm() {
    read -p "$1 (y/n): " answer
    case ${answer:0:1} in
        y|Y )
            return 0
        ;;
        * )
            return 1
        ;;
    esac
}

# Check if Docker is installed
section "Checking Prerequisites"
if ! command -v docker &> /dev/null; then
    error "Docker is not installed. Cannot proceed with cleanup."
    exit 1
fi

if ! command -v docker compose &> /dev/null; then
    error "Docker Compose is not installed. Cannot proceed with cleanup."
    exit 1
fi

# Stop Milvus containers
section "Stopping Milvus Containers"
info "Stopping all Milvus-related containers using Docker Compose"
cd "$(dirname "$0")"

if docker ps | grep -q "ai_politician_milvus\|milvus-etcd\|milvus-minio"; then
    docker compose down
    info "Milvus containers have been stopped"
else
    info "No Milvus containers are currently running"
fi

# Data cleanup section - now with a clearer warning
section "Data Management"
echo -e "${YELLOW}NOTE:${NC} For normal usage, you do NOT need to delete your data."
echo -e "Only delete data if you're sure you want to reset everything and lose all stored information."

if confirm "Do you want to remove all Milvus data? This is NOT recommended for regular cleanup and will DELETE ALL stored data (cannot be undone)"; then
    if confirm "Are you ABSOLUTELY SURE? This will permanently delete ALL database content"; then
        info "Removing Milvus data directories..."
        rm -rf /home/natalie/Databases/ai_politician_milvus/data/*
        rm -rf /home/natalie/Databases/ai_politician_milvus/etcd/*
        rm -rf /home/natalie/Databases/ai_politician_milvus/minio/*
        info "Milvus data has been removed"
    else
        info "Data deletion cancelled. Your data is preserved."
    fi
else
    info "Good choice! Data cleanup skipped. Your database content is preserved for future use."
fi

# Success message
section "Cleanup Complete"
echo -e "${GREEN}Milvus database cleanup is complete!${NC}"
echo -e "All Milvus containers have been stopped but your data is preserved."
echo -e "To restart Milvus, run the setup.sh script."
