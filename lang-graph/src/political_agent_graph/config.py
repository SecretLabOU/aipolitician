"""Configuration for the political agent graph.

This module handles configuring the system for different AI tasks.
"""

from political_agent_graph.local_models import get_model

# Task configurations
TASK_CONFIG = {
    # Task types and the models to use for them
    "analyze_sentiment": {
        "model": "mistral",  # General analytical tasks use base model
        "temperature": 0.1,  # Low temperature for analytical tasks
    },
    "determine_topic": {
        "model": "mistral",
        "temperature": 0.1,
    },
    "decide_deflection": {
        "model": "trump",  # Trump-specific strategic decisions
        "temperature": 0.5,  # Medium temperature for strategic decisions
    },
    "format_response": {
        "model": "trump",  # Trump's speaking style and tone
        "temperature": 0.7,  # Higher temperature for creative response generation
    },
    "generate_policy_stance": {
        "model": "trump",  # Trump's policy positions
        "temperature": 0.5,
    }
}

def get_model_for_task(task_name):
    """Get the appropriate model for a specific task."""
    config = TASK_CONFIG.get(task_name, {
        "model": "mistral",
        "temperature": 0.5,
    })
    
    model_name = config["model"]
    return get_model(model_name)

def get_temperature_for_task(task_name):
    """Get the appropriate temperature for a specific task."""
    config = TASK_CONFIG.get(task_name, {
        "model": "mistral",
        "temperature": 0.5,
    })
    
    return config["temperature"]