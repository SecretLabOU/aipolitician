"""Configuration for the political agent graph.

This module handles configuring the system for different AI tasks.
"""

from political_agent_graph.local_models import get_model

# Active persona tracking
_active_persona = "donald_trump"  # Default to Trump

def select_persona(persona_id):
    """Select the active persona to use."""
    global _active_persona
    _active_persona = persona_id
    print(f"Persona selected: {persona_id}")

def get_active_persona():
    """Get the current active persona ID."""
    return _active_persona

# Model mapping for personas
PERSONA_MODEL_MAP = {
    "donald_trump": "trump",
    "joe_biden": "biden",
    # Add more personas here as they are implemented
}

# Task configurations for each persona
TASK_CONFIG = {
    # General tasks - use the persona-specific model
    "analyze_sentiment": {
        "temperature": 0.1,  # Low temperature for analytical tasks
    },
    "determine_topic": {
        "temperature": 0.1,
    },
    "decide_deflection": {
        "temperature": 0.5,  # Medium temperature for strategic decisions
    },
    "format_response": {
        "temperature": 0.7,  # Higher temperature for creative response generation
    },
    "generate_policy_stance": {
        "temperature": 0.5,
    },

    # Persona-specific overrides - use these if you need different settings per persona
    "donald_trump": {
        "format_response": {
            "temperature": 0.8,  # Trump is more unpredictable in responses
        },
    },
    "joe_biden": {
        "format_response": {
            "temperature": 0.6,  # Biden is more measured in responses
        },
    }
}

def get_model_for_task(task_name):
    """Get the appropriate model for a specific task."""
    # Get the active persona
    persona = _active_persona
    
    # Map persona to model
    model_name = PERSONA_MODEL_MAP.get(persona, "mistral")
    
    # For certain analytical tasks, we might want to use the base model
    if task_name in ["analyze_sentiment", "determine_topic"] and model_name != "mistral":
        # Optional: Uncomment this if you want analytical tasks to use the base model
        # return get_model("mistral")
        pass
    
    return get_model(model_name)

def get_temperature_for_task(task_name):
    """Get the appropriate temperature for a specific task."""
    # Get the active persona
    persona = _active_persona
    
    # Check for persona-specific override for this task
    if persona in TASK_CONFIG and task_name in TASK_CONFIG[persona]:
        return TASK_CONFIG[persona][task_name]["temperature"]
    
    # Otherwise use the default for the task
    if task_name in TASK_CONFIG:
        return TASK_CONFIG[task_name]["temperature"]
    
    # Default fallback
    return 0.5