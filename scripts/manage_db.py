#!/usr/bin/env python3
"""
Database Management Helper Script
This script provides easy commands to start, stop, restart, and check the status of the Milvus database.
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(root_dir))
milvus_dir = root_dir / "src" / "data" / "db" / "milvus"

def run_command(command, cwd=None):
    """Run a shell command and return its output."""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            text=True, 
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, f"Error ({e.returncode}): {e.stderr}"

def check_status():
    """Check the status of Milvus containers."""
    print("Checking Milvus database status...")
    success, output = run_command("docker ps | grep milvus")
    
    if success and output.strip():
        print("✅ Milvus database is running:")
        print(output)
        return True
    else:
        print("❌ Milvus database is not running")
        return False

def start_database():
    """Start the Milvus database."""
    print("Starting Milvus database...")
    
    # First clean up any conflicting containers
    success, output = run_command("docker ps -a | grep milvus", cwd=str(milvus_dir))
    if success and output.strip():
        print("Found existing Milvus containers, cleaning up...")
        run_command("docker rm -f $(docker ps -a | grep milvus | awk '{print $1}')")
    
    # Try to start with docker compose
    success, output = run_command("docker compose up -d", cwd=str(milvus_dir))
    
    if success:
        print("✅ Milvus database started successfully")
        return True
    else:
        print(f"Failed to start with 'docker compose': {output}")
        print("Trying with docker-compose...")
        success, output = run_command("docker-compose up -d", cwd=str(milvus_dir))
        
        if success:
            print("✅ Milvus database started successfully")
            return True
        else:
            print(f"❌ Failed to start Milvus: {output}")
            return False

def stop_database():
    """Stop the Milvus database."""
    print("Stopping Milvus database...")
    
    # Try to stop with docker compose
    success, output = run_command("docker compose down", cwd=str(milvus_dir))
    
    if not success:
        print(f"Failed to stop with 'docker compose': {output}")
        print("Trying with docker-compose...")
        success, output = run_command("docker-compose down", cwd=str(milvus_dir))
    
    if success:
        print("✅ Milvus database stopped successfully")
        return True
    else:
        print(f"❌ Failed to stop Milvus properly: {output}")
        print("Trying to force remove containers...")
        run_command("docker rm -f $(docker ps -a | grep milvus | awk '{print $1}')")
        return False

def restart_database():
    """Restart the Milvus database."""
    print("Restarting Milvus database...")
    stop_database()
    return start_database()

def load_data():
    """Load data into the Milvus database."""
    print("Loading data into Milvus...")
    load_script = milvus_dir / "load_data.py"
    
    if not load_script.exists():
        print(f"❌ Load script not found at {load_script}")
        return False
    
    success, output = run_command(f"python {load_script}", cwd=str(milvus_dir))
    
    if success:
        print("✅ Data loaded successfully")
        return True
    else:
        print(f"❌ Failed to load data: {output}")
        return False

def main():
    """Main entry point for the database management script."""
    parser = argparse.ArgumentParser(description="Manage the Milvus database")
    
    # Add commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start the Milvus database")
    
    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop the Milvus database")
    
    # Restart command
    restart_parser = subparsers.add_parser("restart", help="Restart the Milvus database")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check the status of the Milvus database")
    
    # Load data command
    load_parser = subparsers.add_parser("load", help="Load data into the Milvus database")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle commands
    if args.command == "start":
        start_database()
    elif args.command == "stop":
        stop_database()
    elif args.command == "restart":
        restart_database()
    elif args.command == "status":
        check_status()
    elif args.command == "load":
        load_data()
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 