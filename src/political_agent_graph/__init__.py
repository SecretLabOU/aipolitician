"""Political Agent Graph package.

This package provides a multi-component architecture for simulating political
figures using multiple specialized language models that work together.
"""

# Active persona tracking (simplified)
_active_persona_id = None

def set_active_persona(persona_id: str) -> None:
    """Set the active persona ID.
    
    Args:
        persona_id: The ID of the persona to activate
    """
    global _active_persona_id
    
    # Normalize the persona ID
    persona_id = persona_id.lower().replace(" ", "_")
    _active_persona_id = persona_id

def get_active_persona_id() -> str:
    """Get the active persona ID.
    
    Returns:
        The active persona ID
    
    Raises:
        RuntimeError: If no persona is active
    """
    global _active_persona_id
    
    if _active_persona_id is None:
        raise RuntimeError("No active persona set")
    
    return _active_persona_id

def get_persona_name(persona_id: str) -> str:
    """Get a display name for a persona ID.
    
    Args:
        persona_id: The persona ID
        
    Returns:
        A display name for the persona
    """
    if persona_id.lower() == "donald_trump":
        return "Donald Trump"
    elif persona_id.lower() == "joe_biden":
        return "Joe Biden"
    else:
        # Fallback: convert snake_case to Title Case
        return persona_id.replace("_", " ").title()

def get_persona_party(persona_id: str) -> str:
    """Get the party affiliation for a persona ID.
    
    Args:
        persona_id: The persona ID
        
    Returns:
        The party affiliation
    """
    if persona_id.lower() == "donald_trump":
        return "Republican"
    elif persona_id.lower() == "joe_biden":
        return "Democratic"
    else:
        return "Independent"

# Import other components for convenience
from .graph import run_conversation
from .config import get_config, get_model_for_task