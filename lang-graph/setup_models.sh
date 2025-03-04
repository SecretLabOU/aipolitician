#!/bin/bash
# Simple setup script for Political AI

# Colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

clear
echo -e "${GREEN}Political AI - Quick Setup${NC}"
echo "============================"
echo
echo "This will set up everything you need to talk to AI politicians."
echo

# Create models directory
mkdir -p models

# Check if Ollama is installed
if command -v ollama >/dev/null 2>&1; then
  echo -e "${GREEN}✓${NC} Ollama is already installed"
  
  # Offer to pull models directly
  echo
  echo "Would you like to download the AI model now? (y/n)"
  read -r download_model
  
  if [ "$download_model" = "y" ]; then
    echo
    echo "Downloading Mistral 7B (this may take a few minutes)..."
    ollama pull mistral
    echo -e "${GREEN}✓${NC} Model downloaded successfully!"
  fi
else
  echo -e "${YELLOW}!${NC} Ollama is not installed"
  echo
  echo "Ollama is needed to run the AI models. Install it now? (y/n)"
  read -r install_ollama
  
  if [ "$install_ollama" = "y" ]; then
    echo "Installing Ollama..."
    
    # Install Ollama based on OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
      curl -fsSL https://ollama.com/install.sh | sh
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
      curl -fsSL https://ollama.com/install.sh | sh
    else
      echo -e "${RED}!${NC} Please visit https://ollama.com/download to install Ollama manually"
      exit 1
    fi
    
    echo -e "${GREEN}✓${NC} Ollama installed!"
    
    echo
    echo "Downloading Mistral 7B model (this may take a few minutes)..."
    ollama pull mistral
    echo -e "${GREEN}✓${NC} Model downloaded successfully!"
  else
    echo
    echo -e "${YELLOW}!${NC} You'll need to install Ollama later to use Political AI"
    echo "Visit: https://ollama.com/download"
  fi
fi

# Create simple config file
cat > models/config.json << EOF
{
  "models": {
    "mistral": {
      "type": "ollama",
      "model": "mistral",
      "temperature": 0.7,
      "description": "Mistral 7B - fast and good quality responses"
    }
  },
  "default_model": "mistral"
}
EOF

# Install Python requirements
echo
echo "Installing Python packages..."
pip install -q -r requirements.txt
echo -e "${GREEN}✓${NC} Packages installed"

echo
echo -e "${GREEN}Setup complete!${NC}"
echo
echo "Try it now with: python src/main.py --demo"
echo