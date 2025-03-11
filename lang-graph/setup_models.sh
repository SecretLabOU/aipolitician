#!/bin/bash
# Setup script for Political AI with Lang-graph integration

# Colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Political AI - Lang-graph Setup${NC}"
echo "============================="
echo

# Create models directory
mkdir -p models

# Install Python requirements
echo "Installing Python dependencies..."
pip install -q -r requirements.txt
echo -e "${GREEN}✓${NC} Dependencies installed"

# Create symlinks to the main code for easier access
echo "Setting up integration with main repository..."
PROJECT_ROOT=$(dirname $(dirname $(pwd)))
MAIN_CODE_DIR="${PROJECT_ROOT}"

# Check if main code exists
if [ -f "${MAIN_CODE_DIR}/chat_trump.py" ] && [ -f "${MAIN_CODE_DIR}/chat_biden.py" ]; then
  echo -e "${GREEN}✓${NC} Found main repository code"
else
  echo -e "${YELLOW}!${NC} Could not find main code files at ${MAIN_CODE_DIR}"
  echo "The system will use fallback models instead of the fine-tuned models."
  echo "Make sure you're running this from the lang-graph directory within the aipolitician repo."
fi

# Set up .env file if not exists
if [ ! -f ".env" ]; then
  echo "Creating .env file..."
  touch .env
  echo "# HuggingFace token" >> .env
  echo "# HF_TOKEN=your_token_here" >> .env
  echo -e "${GREEN}✓${NC} Created .env file template"
fi

echo
echo -e "${GREEN}Setup complete!${NC}"
echo "Models will use the main repository's Trump and Biden implementations."
echo "Start talking to politicians: python src/main.py --demo"
echo