#!/bin/bash
# Political Agent System Runner
# A convenience script for running the system

set -e  # Exit on error

# Set up environment colors for output
BLUE='\033[1;34m'
GREEN='\033[1;32m'
RED='\033[1;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print a stylized header
echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}      Political Agent System${NC}"
echo -e "${BLUE}======================================${NC}"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is required but not installed.${NC}"
    exit 1
fi

# Check CUDA availability
echo -e "${YELLOW}Checking GPU availability...${NC}"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Warning: Unable to check CUDA. The system will still work, but may be slower.${NC}"
fi

# Create the models directory if it doesn't exist
if [ ! -d "models" ]; then
    echo -e "${YELLOW}Creating models directory...${NC}"
    mkdir -p models
fi

# Check if we need to install dependencies
if [ "$1" == "--install" ]; then
    echo -e "${GREEN}Installing dependencies...${NC}"
    pip install -r requirements.txt
    shift  # Remove this argument for later processing
fi

# Get the current directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the src directory
cd "$DIR/src"

# Function to display usage
function show_usage {
    echo -e "${GREEN}Usage:${NC}"
    echo -e "  ${YELLOW}./run.sh [--install] [command]${NC}"
    echo -e ""
    echo -e "${GREEN}Commands:${NC}"
    echo -e "  ${YELLOW}list${NC}           List available personas"
    echo -e "  ${YELLOW}chat <persona>${NC} Start a chat with the specified persona"
    echo -e "  ${YELLOW}demo${NC}           Run a demonstration"
    echo -e ""
    echo -e "${GREEN}Examples:${NC}"
    echo -e "  ${YELLOW}./run.sh list${NC}"
    echo -e "  ${YELLOW}./run.sh chat trump${NC}"
    echo -e "  ${YELLOW}./run.sh demo${NC}"
    echo -e ""
}

# Process command
case "$1" in
    list)
        echo -e "${GREEN}Listing available personas...${NC}"
        python3 main.py --list
        ;;
    chat)
        if [ -z "$2" ]; then
            echo -e "${RED}Error: No persona specified.${NC}"
            show_usage
            exit 1
        fi
        echo -e "${GREEN}Starting chat with $2...${NC}"
        python3 main.py --chat "$2"
        ;;
    demo)
        echo -e "${GREEN}Running demonstration...${NC}"
        python3 main.py --demo
        ;;
    help|--help|-h)
        show_usage
        ;;
    "")
        # No command provided, show usage
        show_usage
        ;;
    *)
        echo -e "${RED}Error: Unknown command '$1'${NC}"
        show_usage
        exit 1
        ;;
esac

# Exit cleanly
exit 0 