#!/bin/bash
# Simplified run script for the lang-graph demo

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the application
python src/main.py "$@" 