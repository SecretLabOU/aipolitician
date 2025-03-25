#!/usr/bin/env python3
"""
Visualization utilities for the AI Politician LangGraph system.
"""
import sys
from pathlib import Path
import os
import tempfile
import webbrowser
from typing import Optional

# Add the project root to the Python path
root_dir = Path(__file__).parent.parent.parent.parent.parent.absolute()
sys.path.insert(0, str(root_dir))

from src.models.langgraph.workflow import create_politician_graph

def visualize_graph(output_path: Optional[str] = None, auto_open: bool = True) -> str:
    """
    Generate a visualization of the AI Politician workflow graph.
    
    Args:
        output_path: Path to save the visualization HTML file (optional)
        auto_open: Whether to automatically open the visualization in a browser
        
    Returns:
        Path to the generated HTML file
    """
    # Create the graph
    graph = create_politician_graph()
    
    # If no output path is provided, create a temporary file
    if output_path is None:
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, "ai_politician_graph.html")
    
    # Generate the visualization
    graph.save_graph(output_path)
    
    # Open in browser if requested
    if auto_open:
        webbrowser.open(f"file://{os.path.abspath(output_path)}")
    
    return output_path

if __name__ == "__main__":
    # If run directly, generate and open the visualization
    output_file = visualize_graph()
    print(f"Graph visualization saved to: {output_file}") 