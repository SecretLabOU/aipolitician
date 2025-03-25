#!/bin/bash
# Simplified setup script for the lang-graph demo

set -e  # Exit on error

# Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setup complete! You can now run the demo with:"
echo "python src/main.py --demo"