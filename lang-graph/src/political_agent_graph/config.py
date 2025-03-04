"""Configuration module for the Political Agent Graph system.

This module provides model configuration for different tasks in the agent graph.
"""

import logging

# Setup logging
logger = logging.getLogger(__name__)

# Default model configuration with different models for different tasks
DEFAULT_MODEL_CONFIG = {
    "sentiment_analysis": "local/mixtral",
    "context_extraction": "local/mixtral", 
    "routing": "local/mixtral",
    "tone_generation": "local/trump_mistral",
    "deflection": "local/trump_mistral",
    "response_composition": "local/trump_mistral",
    "fact_checking": "local/mixtral",
    "final_output": "local/trump_mistral",
    "multi_persona": "local/trump_mistral",
    "default": "local/trump_mistral"
}

# Current model configuration
_model_config = DEFAULT_MODEL_CONFIG.copy()

def get_model_for_task(task: str) -> str:
    """Get the appropriate model for a specific task.
    
    Args:
        task: Task name to get model for
        
    Returns:
        Model identifier string
    """
    return _model_config.get(task, _model_config["default"])

def set_model_for_task(task: str, model: str) -> None:
    """Set a specific model for a task.
    
    Args:
        task: Task name to set model for
        model: Model identifier string
    """
    global _model_config
    _model_config[task] = model
    logger.info(f"Set model for {task}: {model}")
    
def reset_model_config() -> None:
    """Reset model configuration to defaults."""
    global _model_config
    _model_config = DEFAULT_MODEL_CONFIG.copy()
    logger.info("Reset model configuration to defaults")

def get_model_config() -> dict:
    """Get the current model configuration."""
    return _model_config.copy()

