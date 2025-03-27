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
                "ENV_ID": env_id,
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
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run AI Politician pipeline with GPU support")
    parser.add_argument("--politicians", default="Donald Trump,Joe Biden",
                        help="Comma-separated list of politicians to process")
    parser.add_argument("--db-path", default=os.path.expanduser("~/political_db"),
                        help="Path to the ChromaDB database")
    parser.add_argument("--output-dir", default=None,
                        help="Path to the output directory containing scraped JSON files")
    parser.add_argument("--env-id", default="nat",
                        help="genv environment ID")
    parser.add_argument("--device-id", type=int, default=1,
                        help="GPU device ID (0: RTX 4060 Ti, 1: RTX 4090)")
    parser.add_argument("--skip-gpu", action="store_true",
                        help="Skip GPU setup (useful for debugging)")
    parser.add_argument("--skip-db", action="store_true",
                        help="Skip loading data into the database")
    
    args = parser.parse_args()
    
    # Set up GPU environment
    gpu_success = True
    if not args.skip_gpu:
        gpu_success = setup_gpu_environment(args.env_id, args.device_id)
        if not gpu_success:
            print("⚠️  Warning: GPU setup failed, continuing without GPU acceleration")
            time.sleep(2)  # Give user time to read the warning
    
    # Run the pipeline
    pipeline_success = run_pipeline(args.politicians, args.db_path)
    
    # Load the data into the database
    db_success = True
    if not args.skip_db and pipeline_success:
        db_success = load_database(args.output_dir, args.db_path)
    elif args.skip_db:
        print("🔵 Skipping database loading as requested")
    elif not pipeline_success:
        print("⚠️ Skipping database loading due to pipeline failure")
    
    # Provide feedback about the result
    if pipeline_success and (db_success or args.skip_db):
        print("\n✅ Pipeline completed successfully!")
        if not args.skip_db and db_success:
            print("✅ Data loaded into database successfully!")
        
        print(f"\n📊 Results can be queried with:")
        print(f"  - To get politician info: python src/data/db/chroma/query.py --query \"Donald Trump\"")
        print(f"  - To get by ID: python src/data/db/chroma/query.py --id <politician-id>")
        print(f"  - To list all politicians: python src/data/db/chroma/query.py --list-all")
        sys.exit(0)
    else:
        print("\n❌ Pipeline process failed!")
        if not pipeline_success:
            print("❌ Scraping pipeline failed")
        if not args.skip_db and not db_success:
            print("❌ Database loading failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 