#!/usr/bin/env python3
"""
Launch script for LangGraph Studio Web UI.
This script starts both the LangGraph server and the Studio Web UI.
"""

import os
import sys
import subprocess
import webbrowser
import time

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

def main():
    print("Starting LangGraph Studio environment...")
    
    # Start the LangGraph server in a separate process
    server_process = subprocess.Popen(
        [sys.executable, "src/run_langgraph_studio.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Wait for server to start
    print("Waiting for server to start...")
    time.sleep(3)
    
    # Launch LangGraph Studio Web UI
    studio_url = "http://localhost:3000"
    print(f"Launching LangGraph Studio Web UI at {studio_url}")
    print(f"Connect to the backend at http://localhost:8000/political-agent")
    
    try:
        webbrowser.open(studio_url)
        
        # Keep the server running until interrupted
        while True:
            output = server_process.stdout.readline()
            if output:
                print(output.strip())
            
            # Check if server process is still running
            if server_process.poll() is not None:
                print("Server process terminated unexpectedly")
                break
                
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        # Clean up the server process
        if server_process.poll() is None:
            server_process.terminate()
            server_process.wait()

if __name__ == "__main__":
    main() 