#!/usr/bin/env python3
"""
AI Politician - High-Performance LLM System

Unified command-line interface for running the AI Politician system in various modes.
Provides advanced GPU optimization, comprehensive monitoring, and production-ready deployments.
"""

import os
import sys
import time
import logging
import argparse
import json
import shutil
import asyncio
import platform
import inspect
import importlib.util
import textwrap
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from datetime import datetime

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.getenv("APP_LOG_FILE", "logs/app.log"))
    ]
)
logger = logging.getLogger("main")

# Create logs directory if it doesn't exist
Path("logs").mkdir(exist_ok=True)

# Version information
VERSION = "1.0.0"
BUILD_DATE = "2023-06-20"  # Update as needed

class RunMode(str, Enum):
    """Available run modes for the application"""
    CLI = "cli"
    TUI = "tui" 
    API = "api"
    CHECK = "check"
    INGEST = "ingest"
    TRAIN = "train"
    BENCHMARK = "benchmark"
    SERVE = "serve"

class Config:
    """Configuration management for the application"""
    
    def __init__(self):
        """Initialize with default configuration"""
        self.config = {}
        self.config_file = Path(os.getenv("CONFIG_FILE", "config.json"))
        self.load_config()
        
        # Environment variables override config file
        self.override_from_env()
    
    def load_config(self):
        """Load configuration from file"""
        try:
            if self.config_file.exists():
                with open(self.config_file, "r") as f:
                    self.config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_file}")
            else:
                logger.info(f"No configuration file found at {self.config_file}, using defaults")
                self.config = self.get_default_config()
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            self.config = self.get_default_config()
    
    def override_from_env(self):
        """Override configuration with environment variables"""
        # Core settings
        if "BASE_MODEL" in os.environ:
            self.config["models"]["base_model"] = os.environ["BASE_MODEL"]
        
        for persona in ["trump", "biden"]:
            env_var = f"{persona.upper()}_MODEL_PATH"
            if env_var in os.environ:
                self.config["models"]["personas"][persona] = os.environ[env_var]
        
        # GPU settings
        if "USE_TENSOR_PARALLEL" in os.environ:
            self.config["gpu"]["tensor_parallel"] = os.environ["USE_TENSOR_PARALLEL"].lower() == "true"
        
        if "TENSOR_PARALLEL_SIZE" in os.environ:
            self.config["gpu"]["tensor_parallel_size"] = int(os.environ["TENSOR_PARALLEL_SIZE"])
        
        if "USE_FLASH_ATTENTION" in os.environ:
            self.config["gpu"]["flash_attention"] = os.environ["USE_FLASH_ATTENTION"].lower() == "true"
            
        # API settings
        if "API_HOST" in os.environ:
            self.config["api"]["host"] = os.environ["API_HOST"]
            
        if "API_PORT" in os.environ:
            self.config["api"]["port"] = int(os.environ["API_PORT"])
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "models": {
                "base_model": "mistralai/Mistral-7B-Instruct-v0.2",
                "personas": {
                    "trump": "nnat03/trump-mistral-adapter",
                    "biden": "nnat03/biden-mistral-adapter"
                },
                "default_persona": "trump",
                "cache_dir": "./models"
            },
            "gpu": {
                "tensor_parallel": False,
                "tensor_parallel_size": 1,
                "flash_attention": True,
                "quantization": "4bit",  # "4bit", "8bit", "awq", or "none"
                "use_vllm": False
            },
            "generation": {
                "max_length": 512,
                "temperature": 0.7
            },
            "rag": {
                "enabled": True,
                "vector_store": "milvus",
                "milvus_host": "localhost",
                "milvus_port": 19530,
                "embedding_model": "all-MiniLM-L6-v2"
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8080,
                "workers": 1,
                "enable_api_key": False,
                "api_key": "",
                "rate_limit": True
            },
            "paths": {
                "logs": "logs",
                "models": "models",
                "data": "data",
                "chat_logs": "chat_logs",
                "sessions": "chat_sessions"
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value with nested key support (e.g., 'models.base_model')"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set a configuration value with nested key support"""
        keys = key.split('.')
        config = self.config
        
        for i, k in enumerate(keys[:-1]):
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self):
        """Save current configuration to file"""
        try:
            with open(self.config_file, "w") as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Saved configuration to {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            return False
    
    def export_env_vars(self):
        """Export configuration as environment variables"""
        # Model settings
        os.environ["BASE_MODEL"] = self.get("models.base_model")
        os.environ["TRUMP_MODEL_PATH"] = self.get("models.personas.trump")
        os.environ["BIDEN_MODEL_PATH"] = self.get("models.personas.biden")
        os.environ["DEFAULT_PERSONA"] = self.get("models.default_persona")
        os.environ["MODEL_CACHE_DIR"] = self.get("models.cache_dir")
        
        # GPU settings
        os.environ["USE_TENSOR_PARALLEL"] = str(self.get("gpu.tensor_parallel")).lower()
        os.environ["TENSOR_PARALLEL_SIZE"] = str(self.get("gpu.tensor_parallel_size"))
        os.environ["USE_FLASH_ATTENTION"] = str(self.get("gpu.flash_attention")).lower()
        os.environ["QUANTIZATION"] = self.get("gpu.quantization")
        os.environ["ENABLE_VLLM"] = str(self.get("gpu.use_vllm")).lower()
        
        # Generation settings
        os.environ["MODEL_MAX_LENGTH"] = str(self.get("generation.max_length"))
        os.environ["MODEL_TEMPERATURE"] = str(self.get("generation.temperature"))
        
        # RAG settings
        os.environ["ENABLE_RAG"] = str(self.get("rag.enabled")).lower()
        os.environ["VECTOR_STORE_TYPE"] = self.get("rag.vector_store")
        os.environ["MILVUS_HOST"] = self.get("rag.milvus_host")
        os.environ["MILVUS_PORT"] = str(self.get("rag.milvus_port"))
        os.environ["EMBEDDING_MODEL"] = self.get("rag.embedding_model")
        
        # API settings
        os.environ["API_HOST"] = self.get("api.host")
        os.environ["API_PORT"] = str(self.get("api.port"))
        os.environ["API_WORKERS"] = str(self.get("api.workers"))
        os.environ["ENABLE_API_KEY"] = str(self.get("api.enable_api_key")).lower()
        os.environ["API_KEY"] = self.get("api.api_key")
        os.environ["RATE_LIMIT_ENABLED"] = str(self.get("api.rate_limit")).lower()
        
        # Path settings
        for key, value in self.get("paths", {}).items():
            os.environ[f"{key.upper()}_DIR"] = value

# Load configuration
config = Config()

def check_dependency(module_name: str) -> bool:
    """Check if a dependency is installed"""
    return importlib.util.find_spec(module_name) is not None

def verify_dependencies(mode: RunMode) -> List[str]:
    """Verify dependencies for a specific mode and return missing ones"""
    # Base dependencies for all modes
    required_deps = ['torch', 'transformers', 'peft']
    
    # Mode-specific dependencies
    if mode == RunMode.API:
        required_deps.extend(['fastapi', 'uvicorn', 'pydantic'])
    elif mode == RunMode.TUI:
        required_deps.extend(['textual'])
    elif mode == RunMode.INGEST:
        required_deps.extend(['sentence_transformers', 'pymilvus'])
    elif mode == RunMode.BENCHMARK:
        required_deps.extend(['numpy', 'matplotlib'])
    elif mode == RunMode.SERVE:
        # For vLLM optimized serving
        required_deps.extend(['vllm', 'ray'])
    
    # Check for GPU acceleration dependencies
    if config.get("gpu.flash_attention"):
        required_deps.append('flash_attn')
    
    if config.get("gpu.tensor_parallel"):
        required_deps.append('accelerate')
    
    # Return missing dependencies
    missing_deps = []
    for dep in required_deps:
        if not check_dependency(dep):
            missing_deps.append(dep)
    
    return missing_deps

def setup_environment():
    """Setup environment variables and paths"""
    # Export config to environment variables
    config.export_env_vars()
    
    # Create required directories
    for path_key, path_value in config.get("paths", {}).items():
        Path(path_value).mkdir(exist_ok=True, parents=True)
    
    # Set CUDA device if specified
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        logger.info(f"Using CUDA devices: {os.environ['CUDA_VISIBLE_DEVICES']}")

def run_cli_interface():
    """Run the CLI interface"""
    try:
        from cli import main as cli_main
        cli_main()
    except ImportError as e:
        logger.error(f"Error importing CLI module: {str(e)}")
        sys.exit(1)

def run_tui_interface():
    """Run the TUI interface"""
    try:
        from cli import main as cli_main
        
        # Create args object with tui flag
        class Args:
            def __init__(self):
                self.tui = True
                self.persona = config.get("models.default_persona")
                self.rag = config.get("rag.enabled")
                self.max_length = config.get("generation.max_length")
                self.temperature = config.get("generation.temperature")
                self.stream = True
                self.debug = logger.getEffectiveLevel() == logging.DEBUG
        
        # Run CLI with TUI flag
        import sys
        sys.argv = [sys.argv[0], "--tui"]  # Force TUI mode
        cli_main()
    except ImportError as e:
        logger.error(f"Error importing TUI module: {str(e)}")
        sys.exit(1)

def run_api_server():
    """Run the API server"""
    try:
        from api import run_server
        run_server()
    except ImportError as e:
        logger.error(f"Error importing API module: {str(e)}")
        sys.exit(1)

def run_ingest_process():
    """Run the ingestion process for RAG data"""
    try:
        from ingest import main as ingest_main
        ingest_main()
    except ImportError as e:
        logger.error(f"Error importing ingestion module: {str(e)}")
        logger.error("Ingestion module not found. Make sure to implement ingest.py for data ingestion.")
        sys.exit(1)

def run_training():
    """Run model fine-tuning"""
    try:
        from train import main as train_main
        train_main()
    except ImportError as e:
        logger.error(f"Error importing training module: {str(e)}")
        logger.error("Training module not found. Make sure to implement train.py for model fine-tuning.")
        sys.exit(1)

def run_benchmarks():
    """Run performance benchmarks"""
    try:
        # Import required modules
        import torch
        import time
        import numpy as np
        from models import model_manager
        
        # Get available personas
        personas = model_manager.get_available_personas()
        
        print(f"\n{'=' * 60}")
        print(f"Running benchmarks on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        print(f"{'=' * 60}")
        
        results = {}
        
        for persona in personas:
            print(f"\nBenchmarking {model_manager.get_display_name(persona)}...")
            
            # Benchmark different batch sizes
            for batch_size in [1, 2, 4]:
                # Skip batch sizes > 1 if not supported
                if batch_size > 1 and not hasattr(model_manager, 'generate_batch'):
                    continue
                
                # Create test prompts
                prompts = [
                    "What is your position on the economy?",
                    "How would you address climate change?",
                    "What is your healthcare policy?",
                    "What are your views on immigration?",
                ] * (batch_size // 4 + 1)
                prompts = prompts[:batch_size]
                
                # Warmup
                print(f"  Warming up batch size {batch_size}...")
                if batch_size == 1:
                    asyncio.run(model_manager.generate_response(
                        persona, prompts[0], None, 128, 0.7, False
                    ))
                else:
                    # Batch generation if supported
                    pass
                
                # Benchmark
                times = []
                tokens = []
                
                for i in range(3):  # Run 3 iterations
                    start_time = time.time()
                    
                    if batch_size == 1:
                        response = asyncio.run(model_manager.generate_response(
                            persona, prompts[0], None, 128, 0.7, False
                        ))
                        num_tokens = len(response.split())
                    else:
                        # Batch generation if supported
                        pass
                    
                    elapsed = time.time() - start_time
                    times.append(elapsed)
                    tokens.append(num_tokens)
                    
                    print(f"    Iteration {i+1}: {elapsed:.2f}s, {num_tokens} tokens")
                
                # Calculate throughput
                avg_time = np.mean(times)
                avg_tokens = np.mean(tokens)
                tokens_per_second = avg_tokens / avg_time
                
                results[f"{persona}_batch{batch_size}"] = {
                    "batch_size": batch_size,
                    "avg_time": avg_time,
                    "avg_tokens": avg_tokens,
                    "tokens_per_second": tokens_per_second
                }
                
                print(f"  Batch size {batch_size}: {tokens_per_second:.2f} tokens/second")
        
        # Print summary
        print(f"\n{'=' * 60}")
        print("Benchmark Summary")
        print(f"{'=' * 60}")
        
        for key, result in results.items():
            print(f"{key}: {result['tokens_per_second']:.2f} tokens/sec with batch size {result['batch_size']}")
        
        # Save results
        results_dir = Path("benchmark_results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"benchmark_{timestamp}.json"
        
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {results_file}")
            
    except ImportError as e:
        logger.error(f"Error importing required modules for benchmarking: {str(e)}")
        sys.exit(1)

def run_optimized_server():
    """Run optimized serving with vLLM if available"""
    try:
        import vllm
        from vllm import LLM, SamplingParams
        from api import app
        import uvicorn
        
        # Initialize vLLM model
        print("Initializing vLLM for high-performance serving...")
        
        # Get model configuration
        base_model = config.get("models.base_model")
        tensor_parallel = config.get("gpu.tensor_parallel")
        tensor_parallel_size = config.get("gpu.tensor_parallel_size")
        
        # Create the vLLM engine
        llm = LLM(
            model=base_model,
            tensor_parallel_size=tensor_parallel_size if tensor_parallel else 1,
            trust_remote_code=False,
            max_model_len=config.get("generation.max_length") * 2
        )
        
        # Store the LLM instance in the app state
        app.state.llm = llm
        
        # Run the server
        uvicorn.run(
            app,
            host=config.get("api.host"),
            port=config.get("api.port"),
            workers=1  # vLLM manages its own parallelism
        )
        
    except ImportError:
        logger.error("vLLM not available. Falling back to standard API server.")
        run_api_server()

def run_system_check():
    """Run system checks and print information"""
    import platform
    
    print("\n===== AI Politician System Check =====\n")
    
    # System information
    print("System Information:")
    print(f"  OS: {platform.system()} {platform.release()}")
    print(f"  Python version: {platform.python_version()}")
    print(f"  CPU: {platform.processor() or 'Unknown'}")
    
    # Directory structure
    print("\nDirectory Structure:")
    for path_name, path_dir in config.get("paths", {}).items():
        path = Path(path_dir)
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"  {path_name}: {path.absolute()} [{status}]")
        
    # Check for models
    print("\nModel Files:")
    model_dir = Path(config.get("models.cache_dir"))
    if model_dir.exists():
        model_files = list(model_dir.glob("**/*.bin"))
        if model_files:
            print(f"  Found {len(model_files)} model files")
            
            # Calculate total size
            total_size = sum(f.stat().st_size for f in model_files) / (1024 ** 3)
            print(f"  Total model size: {total_size:.2f} GB")
        else:
            print("  No model files found")
    else:
        print(f"  Model directory {model_dir} does not exist")
    
    # Check CUDA if torch is available
    print("\nGPU Information:")
    if check_dependency('torch'):
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                
                # Memory info
                memory_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                memory_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                print(f"    Memory allocated: {memory_allocated:.2f} GB")
                print(f"    Memory reserved: {memory_reserved:.2f} GB")
    else:
        print("  PyTorch not installed")
    
    # Check transformers if available
    if check_dependency('transformers'):
        import transformers
        print(f"\nTransformers version: {transformers.__version__}")
    else:
        print("\nTransformers not installed")
    
    # Check FastAPI if available
    if check_dependency('fastapi'):
        import fastapi
        print(f"FastAPI version: {fastapi.__version__}")
    else:
        print("FastAPI not installed")
    
    # Check vector database
    vector_store = config.get("rag.vector_store")
    print(f"\nVector Database: {vector_store}")
    
    if vector_store == "milvus":
        if check_dependency('pymilvus'):
            print("  Milvus client installed")
            
            # Check connection
            try:
                from pymilvus import connections
                connections.connect(
                    alias="default", 
                    host=config.get("rag.milvus_host"), 
                    port=config.get("rag.milvus_port")
                )
                print("  Successfully connected to Milvus")
                connections.disconnect("default")
            except Exception as e:
                print(f"  Failed to connect to Milvus: {str(e)}")
        else:
            print("  Milvus client not installed")
    
    # Check disk space
    try:
        import shutil
        total, used, free = shutil.disk_usage('/')
        print(f"\nDisk space: {free / (1024**3):.1f} GB free / {total / (1024**3):.1f} GB total")
    except:
        print("\nCould not check disk space")
    
    print("\n===== System Check Complete =====\n")

def init_project():
    """Initialize a new project with default configuration"""
    import shutil
    
    logger.info("Initializing new AI Politician project")
    
    # Create default config
    config.config = config.get_default_config()
    
    # Create required directories
    for path_key, path_value in config.get("paths", {}).items():
        Path(path_value).mkdir(exist_ok=True, parents=True)
        logger.info(f"Created directory: {path_value}")
    
    # Save configuration
    config.save()
    
    # Create example ingestion script
    ingest_path = Path("ingest.py")
    
    if not ingest_path.exists():
        example_ingest = """#!/usr/bin/env python3
\"\"\"
Data ingestion script for AI Politician

Processes raw data files and loads them into the vector database.
\"\"\"

import os
import argparse
from pathlib import Path

def main():
    \"\"\"Main ingestion function\"\"\"
    parser = argparse.ArgumentParser(description="Ingest data for AI Politician")
    parser.add_argument("--data-dir", default="data", help="Directory containing data files")
    parser.add_argument("--clear", action="store_true", help="Clear existing data before ingestion")
    args = parser.parse_args()
    
    print(f"Starting data ingestion from {args.data_dir}")
    
    # Implement data processing and vector database ingestion here
    
    print("Ingestion complete")

if __name__ == "__main__":
    main()
"""
        with open(ingest_path, "w") as f:
            f.write(example_ingest)
        
        logger.info(f"Created example ingestion script: {ingest_path}")
    
    logger.info("Project initialization complete")
    print(f"\nProject initialized successfully!")
    print(f"Configuration saved to: {config.config_file}")
    print(f"Run 'python main.py check' to verify your setup")

def create_argument_parser():
    """Create argument parser for the application"""
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    
    description = textwrap.dedent(f"""
    AI Politician - High-Performance LLM System (v{VERSION})
    
    A unified command-line application for running the AI Politician system.
    Supports multiple modes of operation including CLI, API, and system checks.
    """)
    
    parser = ArgumentParser(
        description=description,
        formatter_class=RawDescriptionHelpFormatter
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=False)
    for mode in RunMode:
        mode_group.add_argument(
            f"--{mode.value}", 
            dest="mode",
            action="store_const",
            const=mode.value,
            help=f"Run in {mode.value} mode"
        )
    
    # General options
    parser.add_argument(
        "--init",
        action="store_true",
        help="Initialize a new project with default configuration"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    parser.add_argument(
        "--config",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--update-config",
        action="store_true",
        help="Update config file with current settings"
    )
    
    # GPU options
    gpu_group = parser.add_argument_group("GPU Options")
    gpu_group.add_argument(
        "--device",
        help="CUDA device(s) to use (e.g., '0' or '0,1')"
    )
    
    gpu_group.add_argument(
        "--tensor-parallel",
        action="store_true",
        help="Enable tensor parallelism for multi-GPU"
    )
    
    gpu_group.add_argument(
        "--no-flash-attn",
        action="store_true",
        help="Disable Flash Attention"
    )
    
    # Model options
    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument(
        "--persona",
        choices=["trump", "biden"],
        help="Default persona to use"
    )
    
    model_group.add_argument(
        "--base-model",
        help="Base model to use"
    )
    
    model_group.add_argument(
        "--temperature",
        type=float,
        help="Generation temperature (0.0-1.0)"
    )
    
    model_group.add_argument(
        "--max-length",
        type=int,
        help="Maximum generation length"
    )
    
    # API options
    api_group = parser.add_argument_group("API Options")
    api_group.add_argument(
        "--host",
        help="API host address"
    )
    
    api_group.add_argument(
        "--port",
        type=int,
        help="API port"
    )
    
    api_group.add_argument(
        "--workers",
        type=int,
        help="API worker processes"
    )
    
    api_group.add_argument(
        "--enable-api-key",
        action="store_true",
        help="Enable API key authentication"
    )
    
    api_group.add_argument(
        "--api-key",
        help="API key for authentication"
    )
    
    # RAG options
    rag_group = parser.add_argument_group("RAG Options")
    rag_group.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable RAG for all modes"
    )
    
    rag_group.add_argument(
        "--vector-store",
        choices=["milvus", "faiss"],
        help="Vector store to use"
    )
    
    rag_group.add_argument(
        "--embedding-model",
        help="Embedding model to use"
    )
    
    return parser

def apply_cli_overrides(args):
    """Apply command-line argument overrides to configuration"""
    # GPU options
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    if args.tensor_parallel:
        config.set("gpu.tensor_parallel", True)
    
    if args.no_flash_attn:
        config.set("gpu.flash_attention", False)
    
    # Model options
    if args.persona:
        config.set("models.default_persona", args.persona)
    
    if args.base_model:
        config.set("models.base_model", args.base_model)
    
    if args.temperature is not None:
        config.set("generation.temperature", args.temperature)
    
    if args.max_length is not None:
        config.set("generation.max_length", args.max_length)
    
    # API options
    if args.host:
        config.set("api.host", args.host)
    
    if args.port:
        config.set("api.port", args.port)
    
    if args.workers:
        config.set("api.workers", args.workers)
    
    if args.enable_api_key:
        config.set("api.enable_api_key", True)
    
    if args.api_key:
        config.set("api.api_key", args.api_key)
    
    # RAG options
    if args.no_rag:
        config.set("rag.enabled", False)
    
    if args.vector_store:
        config.set("rag.vector_store", args.vector_store)
    
    if args.embedding_model:
        config.set("rag.embedding_model", args.embedding_model)

def main():
    """Main entry point for the application"""
    # Create and parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    # Initialize project if requested
    if args.init:
        init_project()
        return
    
    # Load alternative config file if specified
    if args.config:
        config.config_file = Path(args.config)
        config.load_config()
    
    # Apply command line overrides
    apply_cli_overrides(args)
    
    # Update config file if requested
    if args.update_config:
        config.save()
        print(f"Configuration updated at {config.config_file}")
        return
    
    # Setup environment
    setup_environment()
    
    # Default to CLI mode if not specified
    mode = args.mode or RunMode.CLI.value
    
    # Check for dependencies first
    try:
        mode_enum = RunMode(mode)
        missing_deps = verify_dependencies(mode_enum)
        
        if missing_deps:
            logger.error(f"Missing dependencies for {mode} mode: {', '.join(missing_deps)}")
            logger.error("Please install missing dependencies before continuing")
            sys.exit(1)
            
    except ValueError:
        logger.error(f"Invalid mode: {mode}")
        logger.error(f"Available modes: {', '.join([m.value for m in RunMode])}")
        sys.exit(1)
    
    # Run in the selected mode
    if mode == RunMode.CLI.value:
        run_cli_interface()
    elif mode == RunMode.TUI.value:
        run_tui_interface()
    elif mode == RunMode.API.value:
        run_api_server()
    elif mode == RunMode.CHECK.value:
        run_system_check()
    elif mode == RunMode.INGEST.value:
        run_ingest_process()
    elif mode == RunMode.TRAIN.value:
        run_training()
    elif mode == RunMode.BENCHMARK.value:
        run_benchmarks()
    elif mode == RunMode.SERVE.value:
        run_optimized_server()

if __name__ == "__main__":
    main() 