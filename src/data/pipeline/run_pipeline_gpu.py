#!/usr/bin/env python3
"""
Political Figure Pipeline with GPU Support

This script runs the AI Politician pipeline with GPU acceleration.
It handles setting up the GPU environment and ensuring proper path configuration.

Usage:
    python run_pipeline_gpu.py [--politicians "Name1,Name2"] [--db-path PATH]
"""

import os
import sys
import argparse
import subprocess

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.insert(0, project_root)

def setup_gpu_environment(env_id="nat", device_id=1):
    """
    Set up the GPU environment using genv
    
    Args:
        env_id: The environment ID to use
        device_id: GPU device ID to use (1 = RTX 4090)
    
    Returns:
        bool: True if setup was successful
    """
    try:
        print("Setting up GPU environment...")
        # Initialize genv
        subprocess.run("eval \"$(genv shell --init)\"", shell=True, check=True)
        
        # Activate environment
        subprocess.run(f"genv activate --id {env_id}", shell=True, check=True)
        
        # Attach GPU (prefer RTX 4090 which is device 1)
        subprocess.run(f"genv attach --device {device_id}", shell=True, check=True)
        
        # Show GPU status
        subprocess.run("nvidia-smi", shell=True, check=True)
        
        print("✅ GPU setup complete!")
        return True
    except Exception as e:
        print(f"❌ Error setting up GPU: {e}")
        print("Continuing without GPU...")
        return False

def run_pipeline(politicians, db_path=None):
    """
    Run the pipeline for the specified politicians
    
    Args:
        politicians: Comma-separated list of politicians
        db_path: Path to the ChromaDB database
        
    Returns:
        bool: True if pipeline completed successfully
    """
    try:
        # Use the pipeline.py script in the same directory
        pipeline_script = os.path.join(current_dir, "pipeline.py")
        cmd = [sys.executable, pipeline_script, "--politicians", politicians]
        
        if db_path:
            cmd.extend(["--db-path", db_path])
            
        print(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        print(f"❌ Error running pipeline: {e}")
        return False

def main():
    """Process command line arguments and run the pipeline"""
    parser = argparse.ArgumentParser(description='Run the political figure pipeline with GPU support')
    parser.add_argument('--politicians', type=str, default="Donald Trump,Joe Biden",
                        help='Comma-separated list of politicians to process')
    parser.add_argument('--db-path', type=str, default=os.path.expanduser("~/political_db"),
                        help='Path to ChromaDB database')
    parser.add_argument('--env-id', type=str, default='nat',
                        help='GPU environment ID for genv')
    parser.add_argument('--device-id', type=int, default=1,
                        help='GPU device ID to use (1 for RTX 4090, 0 for RTX 4060 Ti)')
    
    args = parser.parse_args()
    
    # Set up GPU environment
    setup_gpu_environment(env_id=args.env_id, device_id=args.device_id)
    
    # Run the pipeline
    success = run_pipeline(args.politicians, args.db_path)
    
    if success:
        print("✅ Pipeline completed successfully!")
        print(f"Data has been stored in: {args.db_path}")
        print("\nTo view the results, run:")
        print(f"python {os.path.join(project_root, 'docs/query_database.py')} --query \"Donald Trump\" --db-path {args.db_path}")
        print(f"python {os.path.join(project_root, 'docs/query_database.py')} --query \"Joe Biden\" --db-path {args.db_path}")
    else:
        print("❌ Pipeline failed. Check the logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main() 