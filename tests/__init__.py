"""Test package for PoliticianAI."""

import os
import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set testing environment variable
os.environ["TESTING"] = "true"

# Set test database URL
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["CACHE_DATABASE_URL"] = "sqlite:///:memory:"

# Set test model paths
os.environ["MODELS_DIR"] = str(project_root / "tests" / "models")

# Set test device to CPU
os.environ["DEVICE"] = "cpu"
os.environ["MODEL_PRECISION"] = "float32"

# Disable logging to file in tests
os.environ["LOG_LEVEL"] = "ERROR"
os.environ["LOG_FILE"] = os.devnull
