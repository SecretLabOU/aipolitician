import json
import os
from typing import Dict, List, Optional, Any

# Path to the personas.json file
_PERSONAS_PATH = os.path.join(os.path.dirname(__file__), "personas.json")

# Store the current active persona
_active_persona: Optional[str] = None
# Cache for loaded personas
_personas_cache: Optional[Dict[str, Any]] = None

def load_personas() -> Dict[str, Any]:
    """
    Load all personas from the personas.json file.
    
    Returns:
        Dict[str, Any]: A dictionary mapping persona names to their configurations.
    """
    global _personas_cache
    
    if _personas_cache is None:
        try:
            with open(_PERSONAS_PATH, 'r') as f:
                _personas_cache = json.load(f)
        except Exception as e:
            print(f"Error loading personas from {_PERSONAS_PATH}: {e}")
            _personas_cache = {}
    
    return _personas_cache

def get_persona_names() -> List[str]:
    """
    Get a list of all available persona names.
    
    Returns:
        List[str]: List of persona names.
    """
    personas = load_personas()
    return list(personas.keys())

def get_persona(name: str) -> Optional[Dict[str, Any]]:
    """
    Get the configuration for a specific persona.
    
    Args:
        name (str): The name of the persona to retrieve.
    
    Returns:
        Optional[Dict[str, Any]]: The persona configuration, or None if not found.
    """
    personas = load_personas()
    return personas.get(name)

def set_active_persona(name: str) -> bool:
    """
    Set the active persona.
    
    Args:
        name (str): The name of the persona to set as active.
    
    Returns:
        bool: True if the persona was successfully set, False otherwise.
    """
    global _active_persona
    
    personas = load_personas()
    if name in personas:
        _active_persona = name
        return True
    return False

def get_active_persona() -> Optional[str]:
    """
    Get the name of the currently active persona.
    
    Returns:
        Optional[str]: The name of the active persona, or None if no persona is active.
    """
    return _active_persona

def get_active_persona_config() -> Optional[Dict[str, Any]]:
    """
    Get the configuration of the currently active persona.
    
    Returns:
        Optional[Dict[str, Any]]: The configuration of the active persona, or None if no persona is active.
    """
    if _active_persona is None:
        return None
    
    return get_persona(_active_persona)

def get_persona_trait(name: str, trait: str) -> Optional[Any]:
    """
    Get a specific trait from a persona's configuration.
    
    Args:
        name (str): The name of the persona.
        trait (str): The trait to retrieve.
    
    Returns:
        Optional[Any]: The value of the trait, or None if the persona or trait doesn't exist.
    """
    persona = get_persona(name)
    if persona is None:
        return None
    
    return persona.get(trait)

def get_active_persona_trait(trait: str) -> Optional[Any]:
    """
    Get a specific trait from the active persona's configuration.
    
    Args:
        trait (str): The trait to retrieve.
    
    Returns:
        Optional[Any]: The value of the trait, or None if no active persona or trait doesn't exist.
    """
    if _active_persona is None:
        return None
    
    return get_persona_trait(_active_persona, trait)

def initialize_personas() -> None:
    """
    Initialize the persona system. This should be called at the start of the application.
    
    If no active persona is set, it will set the first available persona as active.
    """
    global _active_persona
    
    personas = load_personas()
    if not personas:
        print("Warning: No personas available.")
        return
    
    if _active_persona is None and personas:
        # Set the first persona as active if none is set
        _active_persona = list(personas.keys())[0]
        print(f"Initialized with default persona: {_active_persona}")

