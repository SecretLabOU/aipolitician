#!/usr/bin/env python3
"""
System compatibility checker for the AI Politician LangGraph system.
Run this script to check if your system is compatible with the models.
"""
import sys
import platform
import os
import subprocess
from pathlib import Path

def print_section(title):
    """Print a section title."""
    print("\n" + "=" * 40)
    print(f" {title}")
    print("=" * 40)

def run_command(command):
    """Run a shell command and return the output."""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr.strip()}"
    except Exception as e:
        return f"Error: {str(e)}"

def check_python():
    """Check Python version and environment."""
    print_section("Python Environment")
    print(f"Python version: {platform.python_version()}")
    print(f"Python executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    
    # Check if we're in a virtual environment
    in_venv = sys.prefix != sys.base_prefix
    print(f"Virtual environment: {'Yes' if in_venv else 'No'}")

def check_required_packages():
    """Check if required packages are installed."""
    print_section("Required Packages")
    
    required_packages = [
        "torch",
        "transformers",
        "pydantic",
        "langgraph",
        "langchain",
        "bitsandbytes",
        "accelerate",
        "fastapi",
        "uvicorn"
    ]
    
    for package in required_packages:
        try:
            spec = __import__(package)
            if hasattr(spec, "__version__"):
                version = spec.__version__
            elif hasattr(spec, "version"):
                version = spec.version.__version__
            else:
                version = "Unknown version"
            print(f"✅ {package}: {version}")
        except ImportError:
            print(f"❌ {package}: Not installed")

def check_cuda():
    """Check CUDA availability and version."""
    print_section("CUDA and GPU")
    
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {'Yes' if cuda_available else 'No'}")
        
        if cuda_available:
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # Convert to GB
                print(f"GPU {i}: {gpu_name} ({gpu_mem:.2f} GB)")
        
        # Try to compile a simple operation to check if CUDA works properly
        if cuda_available:
            x = torch.randn(10, 10).cuda()
            y = x + x
            print("CUDA test: Success")
    
    except Exception as e:
        print(f"Error checking CUDA: {str(e)}")

def check_model_compatibility():
    """Check compatibility with specific models."""
    print_section("Model Compatibility")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch
        
        # Check if we can load a small model
        print("Testing model loading...")
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        try:
            # Check tokenizer loading
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            print(f"✅ Tokenizer loaded successfully for {model_id}")
            
            # Check if 4-bit loading works
            if torch.cuda.is_available():
                try:
                    from bitsandbytes.nn import Linear4bit
                    
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                    
                    # Load a small part of the model to test
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        quantization_config=bnb_config,
                        device_map="auto",
                        torch_dtype=torch.float16,
                        max_memory={0: "2GiB"},
                        attn_implementation="eager"
                    )
                    print(f"✅ 4-bit model loading works")
                except Exception as e:
                    print(f"❌ 4-bit model loading failed: {str(e)}")
        except Exception as e:
            print(f"❌ Model loading failed: {str(e)}")
    
    except ImportError as e:
        print(f"Cannot check model compatibility: {str(e)}")

def main():
    """Run all checks."""
    print("AI Politician System Compatibility Check")
    print("=======================================")
    
    check_python()
    check_required_packages()
    check_cuda()
    check_model_compatibility()
    
    print("\nCompatibility check complete!")
    print("If all checks passed, your system should be compatible with the AI Politician system.")
    print("If you encountered errors, check the requirements and installation instructions.")

if __name__ == "__main__":
    main() 