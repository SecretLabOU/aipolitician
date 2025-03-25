#!/usr/bin/env python3
"""
Diagnostic script to help identify why the data loading pipeline is failing.
"""

import os
import sys
import traceback
from pathlib import Path
import importlib.util
import inspect

# Add project root to path
root_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(root_dir))

def check_module_exists(module_path):
    """Check if a module exists and can be imported"""
    try:
        spec = importlib.util.find_spec(module_path)
        exists = spec is not None
        print(f"✅ Module '{module_path}' {'exists' if exists else 'does not exist'}")
        return exists
    except ModuleNotFoundError:
        print(f"❌ Module '{module_path}' not found")
        return False
    except Exception as e:
        print(f"❌ Error checking for module '{module_path}': {e}")
        return False

def check_file_exists(file_path):
    """Check if a file exists"""
    path = Path(file_path)
    exists = path.exists()
    print(f"{'✅' if exists else '❌'} File '{file_path}' {'exists' if exists else 'does not exist'}")
    return exists

def try_import(module_name):
    """Try to import a module and report details about it"""
    try:
        print(f"Trying to import '{module_name}'...")
        module = __import__(module_name, fromlist=['*'])
        print(f"✅ Successfully imported '{module_name}'")
        
        # Print some info about the module
        print(f"Module location: {module.__file__}")
        print(f"Available functions/classes:")
        for name, obj in inspect.getmembers(module):
            if not name.startswith("_"):  # Skip private/internal objects
                obj_type = type(obj).__name__
                print(f"  - {name} ({obj_type})")
        
        return module
    except ImportError as e:
        print(f"❌ Import error for '{module_name}': {e}")
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"❌ Unexpected error importing '{module_name}': {e}")
        traceback.print_exc()
        return None

def main():
    """Run diagnostic tests"""
    print("=" * 50)
    print("AI Politician Pipeline Diagnostics")
    print("=" * 50)
    
    # Check important directories
    print("\nChecking key directories...")
    directories = [
        "src/data",
        "src/data/db",
        "src/data/db/milvus",
        "src/data/db/milvus/scripts",
        "src/data/pipeline",
        "src/data/scraper",
    ]
    
    for directory in directories:
        exists = Path(directory).is_dir()
        print(f"{'✅' if exists else '❌'} Directory '{directory}' {'exists' if exists else 'does not exist'}")
    
    # Check important files
    print("\nChecking key files...")
    files = [
        "src/data/pipeline/pipeline.py",
        "src/data/db/milvus/scripts/schema.py",
        "src/data/scraper/politician_scraper.py",
        "scripts/load_milvus_data.py",
    ]
    
    for file in files:
        check_file_exists(file)
    
    # Try importing key modules
    print("\nTrying to import key modules...")
    
    pipeline_module = try_import("src.data.pipeline.pipeline")
    schema_module = try_import("src.data.db.milvus.scripts.schema")
    scraper_module = try_import("src.data.scraper.politician_scraper")
    
    # Check if run_pipeline function exists and what it requires
    if pipeline_module and hasattr(pipeline_module, "run_pipeline"):
        print("\nExamining run_pipeline function...")
        run_pipeline = getattr(pipeline_module, "run_pipeline")
        
        # Get the function signature
        sig = inspect.signature(run_pipeline)
        print(f"Function signature: {sig}")
        
        print("Parameters:")
        for name, param in sig.parameters.items():
            print(f"  - {name}: {param.annotation}")
    else:
        print("\n❌ run_pipeline function not found in pipeline module")
    
    # Check the database connection
    if schema_module and hasattr(schema_module, "connect_to_milvus"):
        print("\nTrying to connect to Milvus...")
        try:
            connect_to_milvus = getattr(schema_module, "connect_to_milvus")
            result = connect_to_milvus()
            print(f"{'✅' if result else '❌'} Connection to Milvus {'successful' if result else 'failed'}")
        except Exception as e:
            print(f"❌ Error connecting to Milvus: {e}")
            traceback.print_exc()
    
    print("\nDiagnostic checks completed.")

if __name__ == "__main__":
    main() 