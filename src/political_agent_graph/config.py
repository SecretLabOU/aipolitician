"""Configuration for the political agent graph.

This module handles configuration settings for the Political Agent Graph,
including which models to use for which tasks and other parameters.
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path

# Import the local models module for model instantiation
from .local_models import SimpleModel, TrumpLLM, BidenLLM

# Default configuration
_default_config = {
    "models": {
        # Task-specific model assignments
        "analyze_sentiment": "simple",
        "determine_topic": "simple",
        "decide_deflection": "simple",
        "generate_policy_stance": "persona",  # Uses the persona-specific model
        "format_response": "persona",  # Uses the persona-specific model
        
        # Model parameters
        "temperature": {
            "analyze_sentiment": 0.1,  # Low temperature for more deterministic outputs
            "determine_topic": 0.1,
            "decide_deflection": 0.3,
            "generate_policy_stance": 0.7,  # Higher temperature for more creative policy positions
            "format_response": 0.8   # Higher temperature for more varied responses
        }
    },
    
    # System parameters
    "system": {
        "log_level": os.environ.get("LOG_LEVEL", "INFO"),
        "debug_mode": os.environ.get("DEBUG_MODE", "false").lower() == "true"
    }
}

# Current active configuration
_current_config = _default_config.copy()

def get_config(override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get the current configuration.
    
    Args:
        override: Optional configuration overrides
        
    Returns:
        The current configuration with any overrides applied
    """
    global _current_config
    
    # Apply overrides if provided
    if override:
        # Deep merge the override into the current config
        # (simplified implementation - would need recursive merging for nested dicts)
        for key, value in override.items():
            _current_config[key] = value
    
    return _current_config

def get_model_for_task(task: str) -> Any:
    """Get the appropriate model for a specific task.
    
    Args:
        task: The task name (e.g., "analyze_sentiment")
        
    Returns:
        A model instance appropriate for the task
    """
    config = get_config()
    model_type = config["models"].get(task, "simple")
    
    # Handle persona-specific models
    if model_type == "persona":
        # Get the active persona ID
        from . import get_active_persona_id
        persona_id = get_active_persona_id()
        
        # Return the appropriate persona-specific model
        if persona_id == "donald_trump":
            return TrumpLLM()
        elif persona_id == "joe_biden":
            return BidenLLM()
        else:
            # Fallback to simple model
            return SimpleModel(persona_id)
    
    # Return a simple model for other tasks
    return SimpleModel("assistant")

def get_temperature_for_task(task: str) -> float:
    """Get the temperature setting for a specific task.
    
    Args:
        task: The task name (e.g., "analyze_sentiment")
        
    Returns:
        The temperature value for the task
    """
    config = get_config()
    return config["models"]["temperature"].get(task, 0.7)