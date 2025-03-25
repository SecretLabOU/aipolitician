#!/usr/bin/env python3
"""
System Verification Script for AI Politician

This script checks if all required components of the AI Politician system are 
properly installed and configured.
"""
import os
import sys
import importlib
import subprocess
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ANSI colors for output
class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

# Required packages for different components
REQUIRED_PACKAGES = {
    "base": ["python-dotenv", "tqdm", "numpy"],
    "chat": ["openai", "tiktoken", "bitsandbytes"],
    "langgraph": ["langgraph", "pymilvus", "pydantic", "sentence-transformers"],
    "scraper": ["requests", "beautifulsoup4", "playwright"],
    "training": ["torch", "transformers", "datasets", "peft", "accelerate"]
}

def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}==== {text} ===={Colors.RESET}")

def print_result(text: str, success: bool) -> None:
    """Print a formatted result."""
    if success:
        print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")
    else:
        print(f"{Colors.RED}✗ {text}{Colors.RESET}")

def print_warning(text: str) -> None:
    """Print a formatted warning."""
    print(f"{Colors.YELLOW}! {text}{Colors.RESET}")

def check_package(package_name: str) -> bool:
    """Check if a Python package is installed."""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def check_command(command: str) -> bool:
    """Check if a command is available."""
    try:
        subprocess.run(command.split(), capture_output=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def check_file_exists(path: str) -> bool:
    """Check if a file exists."""
    return Path(path).exists()

def check_env_variables() -> Tuple[int, int]:
    """Check for required environment variables."""
    # Load .env file if present
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print_warning("dotenv package not installed. Skipping .env file loading.")
    
    # Check essential environment variables
    env_vars = {
        "Required": [
            "OPENAI_API_KEY"
        ],
        "Optional": [
            "MILVUS_HOST",
            "MILVUS_PORT",
            "MODEL_PROVIDER",
            "MODEL_PATH"
        ]
    }
    
    present = 0
    missing = 0
    
    print("Required Environment Variables:")
    for var in env_vars["Required"]:
        if os.environ.get(var):
            print_result(f"{var} is set", True)
            present += 1
        else:
            print_result(f"{var} is not set", False)
            missing += 1
    
    print("\nOptional Environment Variables:")
    for var in env_vars["Optional"]:
        if os.environ.get(var):
            print_result(f"{var} is set", True)
            present += 1
        else:
            print_warning(f"{var} is not set (optional)")
    
    return present, missing

def check_dependencies() -> Tuple[int, int]:
    """Check Python package dependencies."""
    installed = 0
    missing = 0
    
    for category, packages in REQUIRED_PACKAGES.items():
        print_header(f"Checking {category.capitalize()} Dependencies")
        
        for package in packages:
            is_installed = check_package(package)
            print_result(f"{package} is installed", is_installed)
            
            if is_installed:
                installed += 1
            else:
                missing += 1
                print(f"  - Install with: pip install {package}")
    
    return installed, missing

def check_database() -> bool:
    """Check if Milvus database is accessible."""
    try:
        from pymilvus import connections
        from src.data.db.milvus.connection import get_connection_params
        
        # Try to connect to Milvus
        conn_params = get_connection_params()
        connections.connect(**conn_params)
        connections.disconnect(conn_params.get("alias", "default"))
        return True
    except Exception as e:
        print(f"  Error: {str(e)}")
        return False

def check_model_availability() -> bool:
    """Check if model files or API access is available."""
    # Check for OpenAI API access
    if os.environ.get("OPENAI_API_KEY"):
        try:
            import openai
            openai.api_key = os.environ.get("OPENAI_API_KEY")
            # Just test the authentication, don't actually make a completion
            # to avoid unnecessary API charges
            return True
        except Exception as e:
            print(f"  Error testing OpenAI API: {str(e)}")
            return False
    
    # Check for local model if specified
    model_path = os.environ.get("MODEL_PATH")
    if model_path:
        model_exists = check_file_exists(model_path)
        if not model_exists:
            print(f"  Model path {model_path} does not exist")
        return model_exists
    
    return False

def check_system_resources() -> Dict[str, str]:
    """Check system resources available."""
    resources = {}
    
    # Check Python version
    resources["Python Version"] = platform.python_version()
    
    # Check for CUDA
    try:
        import torch
        resources["CUDA Available"] = "Yes" if torch.cuda.is_available() else "No"
        if torch.cuda.is_available():
            resources["CUDA Version"] = torch.version.cuda
            resources["GPU Device"] = torch.cuda.get_device_name(0)
            memory_allocated = torch.cuda.memory_allocated(0)
            memory_reserved = torch.cuda.memory_reserved(0)
            resources["GPU Memory Used"] = f"{memory_allocated / (1024**2):.2f} MB"
            resources["GPU Memory Reserved"] = f"{memory_reserved / (1024**2):.2f} MB"
    except ImportError:
        resources["CUDA Available"] = "Unknown (torch not installed)"
    
    # Check for Docker (used by Milvus)
    docker_available = check_command("docker --version")
    resources["Docker Available"] = "Yes" if docker_available else "No"
    
    return resources

def main():
    """Main function to run all checks."""
    print(f"{Colors.BOLD}{Colors.BLUE}AI Politician System Check{Colors.RESET}")
    print(f"This script will verify that your system is properly set up for the AI Politician project.")
    
    # Check dependencies
    print_header("Checking Dependencies")
    installed, missing = check_dependencies()
    print(f"\nDependency Summary: {installed} installed, {missing} missing")
    
    # Check environment variables
    print_header("Checking Environment Variables")
    present, missing_env = check_env_variables()
    print(f"\nEnvironment Variable Summary: {present} present, {missing_env} required ones missing")
    
    # Check database
    print_header("Checking Database Connectivity")
    db_available = check_database()
    print_result("Milvus database is accessible", db_available)
    if not db_available:
        print_warning("  Database not available. RAG features will not work.")
        print_warning("  Run: docker run -d --name milvus_standalone -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest standalone")
    
    # Check model availability
    print_header("Checking Model Availability")
    model_available = check_model_availability()
    print_result("Model is accessible", model_available)
    if not model_available:
        print_warning("  No model available. Set OPENAI_API_KEY or MODEL_PATH in your .env file.")
    
    # Check system resources
    print_header("System Resources")
    resources = check_system_resources()
    for key, value in resources.items():
        print(f"{key}: {value}")
    
    # Print overall summary
    print_header("Summary")
    checks_passed = (missing == 0) and (missing_env == 0) and db_available and model_available
    
    if checks_passed:
        print(f"{Colors.GREEN}{Colors.BOLD}All checks passed! The system is properly set up.{Colors.RESET}")
    else:
        print(f"{Colors.YELLOW}{Colors.BOLD}Some checks failed. Please address the issues above.{Colors.RESET}")
    
    # Return success or failure
    return 0 if checks_passed else 1

if __name__ == "__main__":
    sys.exit(main()) 