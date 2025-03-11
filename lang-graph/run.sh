#!/bin/bash
# Run script for Political AI with Lang-graph

# Get the absolute path to the project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Add the project root to PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Check if main code exists 
if [ -f "${PROJECT_ROOT}/chat_trump.py" ] && [ -f "${PROJECT_ROOT}/chat_biden.py" ]; then
  echo "Found main repository code at ${PROJECT_ROOT}"
else
  echo "Warning: Could not find main code files at ${PROJECT_ROOT}"
  echo "The system will use fallback models instead of the fine-tuned models."
fi

# Run the main script with all arguments passed to this script
cd "$SCRIPT_DIR"
python src/main.py "$@" 