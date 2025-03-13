"""
Script to run the LangGraph Studio Web server.

This script runs the LangGraph Studio Web server, which provides a UI for
visualizing and interacting with the AI Politician graph.
"""

import os
import subprocess
import sys
from pathlib import Path

def main():
    """Run the LangGraph Studio Web server."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent
    
    # Change to the project root directory
    os.chdir(project_root)
    
    # Run the LangGraph development server
    subprocess.run(
        ["langgraph", "dev", "--port", "8000"],
        check=True,
    )

if __name__ == "__main__":
    main()
