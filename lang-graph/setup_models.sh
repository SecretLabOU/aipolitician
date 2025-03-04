#!/bin/bash
# Setup script for Political AI with Trump model

# Colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Political AI - Setup${NC}"
echo "====================="
echo

# Create models directory
mkdir -p models

# Check if Ollama is installed for base models
if command -v ollama >/dev/null 2>&1; then
  echo -e "${GREEN}✓${NC} Ollama is already installed"
  
  echo "Downloading Mistral model for general tasks..."
  ollama pull mistral
  echo -e "${GREEN}✓${NC} Mistral model ready"
else
  echo -e "${YELLOW}!${NC} Ollama is not installed"
  echo "Installing Ollama for base models..."
  
  # Install Ollama based on OS
  if [[ "$OSTYPE" == "darwin"* ]]; then
    curl -fsSL https://ollama.com/install.sh | sh
  elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    curl -fsSL https://ollama.com/install.sh | sh
  else
    echo -e "${RED}!${NC} Please visit https://ollama.com/download to install Ollama manually"
    exit 1
  fi
  
  echo -e "${GREEN}✓${NC} Ollama installed"
  echo "Downloading Mistral model..."
  ollama pull mistral
  echo -e "${GREEN}✓${NC} Mistral model ready"
fi

# Check for Trump model
TRUMP_MODEL_PATH="../../fine_tuned_trump_mistral/model.gguf"
if [ -f "$TRUMP_MODEL_PATH" ]; then
  echo -e "${GREEN}✓${NC} Trump model found"
else
  echo -e "${YELLOW}!${NC} Trump model not found at $TRUMP_MODEL_PATH"
  echo "You will need to add the model file to use Trump's speech patterns."
fi

# Install Python requirements
echo "Installing Python dependencies..."
pip install -q -r requirements.txt
echo -e "${GREEN}✓${NC} Dependencies installed"

echo
echo -e "${GREEN}Setup complete!${NC}"
echo "Start talking to politicians: python src/main.py --demo"
echo