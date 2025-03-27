#!/usr/bin/env python3
"""
AI Politician Pipeline with GPU Acceleration

This script runs the AI Politician pipeline with proper path configuration and GPU support.
It handles setting up the GPU environment, configuring the Python path, and running the 
pipeline for selected politicians.

Usage:
    python3 run_pipeline_gpu.py
    python3 run_pipeline_gpu.py --politicians "Donald Trump,Joe Biden"
    python3 run_pipeline_gpu.py --device-id 0  # Use a different GPU
    python3 run_pipeline_gpu.py --skip-gpu    # Skip GPU setup entirely
    python3 run_pipeline_gpu.py --skip-db     # Skip loading data into the database
"""

import os
import sys
import argparse
import subprocess
import tempfile
import time
import shutil
import pathlib

# Define paths
# Get the path to the current file
CURRENT_DIR = pathlib.Path(__file__).parent.absolute()
# Path to the scraper output directory
OUTPUT_DIR_PAGES = CURRENT_DIR.parent.parent / "scraper" / "politician_crawler" / "output"
# Path to the ChromaDB directory
OUTPUT_DIR_CHROMA = CURRENT_DIR.parent / "db" / "chroma" / "data"

def setup_gpu_environment(env_id="nat", device_id=1):
    """
    Set up GPU environment using genv.
    
    Args:
        env_id: The genv environment ID (default: "nat")
        device_id: The GPU device ID (0: RTX 4060 Ti, 1: RTX 4090)
        
    Returns:
        bool: True if successful, False otherwise
    """
    print("🔌 Setting up GPU environment...")
    
    # Create a temporary shell script to set up the GPU environment
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        script_path = f.name
        f.write("""#!/bin/bash
# Initialize genv
source ~/.bashrc
if command -v genv &> /dev/null; then
    echo "✅ genv is available"
else
    echo "❌ genv command not found"
    exit 1
fi

# Create and activate environment
genv init ${ENV_ID} || true
genv shell ${ENV_ID}

# Attach GPU
genv gpu attach ${DEVICE_ID}

# Check GPU status
echo "📊 GPU Status:"
nvidia-smi
""")
    
    try:
        # Make script executable
        os.chmod(script_path, 0o755)
        
        # Run the script with environment variables
        result = subprocess.run(
            ["bash", script_path],
            env={
                **os.environ,
                "ENV_ID": str(env_id),
                "DEVICE_ID": str(device_id)
            },
            capture_output=True,
            text=True
        )
        
        # Clean up the temporary script
        os.unlink(script_path)
        
        # Check if successful
        if result.returncode != 0:
            print(f"❌ GPU setup failed with exit code {result.returncode}")
            print(f"Error output: {result.stderr}")
            return False
            
        print(result.stdout)
        print("✅ GPU environment set up successfully")
        return True
    
    except Exception as e:
        print(f"❌ Error setting up GPU environment: {e}")
        # Clean up the temporary script if it exists
        if os.path.exists(script_path):
            os.unlink(script_path)
        return False

def run_pipeline(politicians, db_path=None):
    """
    Run the pipeline for the specified politicians.
    
    Args:
        politicians: Comma-separated list of politicians to process
        db_path: Path to the ChromaDB database (optional)
        
    Returns:
        bool: True if the pipeline ran successfully, False otherwise
    """
    # Get the path to the pipeline script relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_script = os.path.join(script_dir, "pipeline.py")
    
    # Construct the command
    cmd = [sys.executable, pipeline_script, "--politicians", politicians]
    if db_path:
        cmd.extend(["--db-path", db_path])
    
    print(f"🚀 Running pipeline for politicians: {politicians}")
    print(f"📋 Command: {' '.join(cmd)}")
    
    try:
        # Set PYTHONPATH to include project root in the subprocess environment
        env = os.environ.copy()
        
        # Add the src directory to PYTHONPATH
        src_dir = os.path.abspath(os.path.join(script_dir, "../.."))
        project_root = os.path.abspath(os.path.join(src_dir, ".."))
        
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{src_dir}:{project_root}:{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = f"{src_dir}:{project_root}"
        
        print(f"🔧 PYTHONPATH: {env['PYTHONPATH']}")
        
        # Run the pipeline
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        
        # Check for success
        if result.returncode == 0:
            print("✅ Pipeline completed successfully!")
            print(result.stdout)
            return True
        else:
            print(f"❌ Pipeline failed with exit code {result.returncode}")
            print("Error output:")
            print(result.stderr)
            print("Standard output:")
            print(result.stdout)
            return False
    
    except Exception as e:
        print(f"❌ Error running pipeline: {e}")
        return False

def load_database(output_dir, db_path):
    """
    Load the data into the ChromaDB database.
    
    Args:
        output_dir: Directory containing the scraped data JSON files
        db_path: Path to the ChromaDB database
        
    Returns:
        bool: True if the database was loaded successfully, False otherwise
    """
    try:
        print(f"📥 Loading data into database at {db_path}...")
        
        # If the database path exists, delete it first to start fresh
        if os.path.exists(db_path):
            print(f"🗑️ Removing existing database at {db_path}...")
            shutil.rmtree(db_path, ignore_errors=True)
            print(f"✅ Existing database removed")
        
        # Import the loader module
        try:
            from src.data.db.chroma.loader import load_database as load_db
            
            # Get the output directory from the politician crawler
            if not output_dir:
                # Default to the politician_crawler output directory
                output_dir = os.path.abspath(os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "../../data/scraper/politician_crawler/output"
                ))
                
            print(f"📂 Using output directory: {output_dir}")
            
            # Load the database
            stats = load_db(output_dir, db_path, verbose=True)
            
            if stats["loaded"] > 0:
                print(f"✅ Successfully loaded {stats['loaded']} entries into the database")
                return True
            else:
                print(f"⚠️ No entries were loaded into the database")
                return False
            
        except ImportError as e:
            print(f"❌ Error importing database loader: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Error loading database: {e}")
        return False

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Run the political figure pipeline with GPU support")
    parser.add_argument("--politicians", help="Comma-separated list of politicians to process", default="Donald Trump,Joe Biden")
    parser.add_argument("--skip-gpu", action="store_true", help="Skip GPU environment setup")
    parser.add_argument("--gpu-env-id", type=int, default=10, help="GPU environment ID")
    parser.add_argument("--skip-db", action="store_true", help="Skip loading the database")
    parser.add_argument("--force-db-load", action="store_true", help="Force loading the database even if pipeline fails")
    parser.add_argument("--allow-partial", action="store_true", help="Return success if at least one politician is processed successfully")
    args = parser.parse_args()

    print(f"Running AI Politician Pipeline with GPU support")
    
    # Setup GPU environment if needed
    if not args.skip_gpu:
        setup_gpu_environment(args.gpu_env_id)
    else:
        print("Skipping GPU environment setup")
    
    # Run the pipeline for the specified politicians
    pipeline_cmd = [
        "python3", "-m", "src.data.pipeline.pipeline",
        "--politicians", args.politicians,
        "--db-path", str(OUTPUT_DIR_CHROMA.parent)
    ]
    
    if args.allow_partial:
        pipeline_cmd.append("--allow-partial")
    
    print(f"Running command: {' '.join(pipeline_cmd)}")
    
    pipeline_result = subprocess.run(pipeline_cmd)
    pipeline_success = pipeline_result.returncode == 0
    
    # Load the database if the pipeline succeeded or if we want to force it
    if (pipeline_success or args.force_db_load) and not args.skip_db:
        print("Loading data into the database...")
        load_database(OUTPUT_DIR_PAGES, OUTPUT_DIR_CHROMA.parent)
    elif args.skip_db:
        print("Skipping database loading as requested")
    elif not pipeline_success:
        print("Pipeline failed, skipping database loading. Use --force-db-load to load anyway.")
    
    return pipeline_result.returncode

if __name__ == "__main__":
    main() 