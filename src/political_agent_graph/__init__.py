"""Political Agent Graph package.

This package provides a multi-component architecture for simulating political
figures using multiple specialized language models that work together.
"""

# Define personae
personas = {
    "donald_trump": {
        "name": "Donald Trump",
        "party": "Republican",
        "bio": "45th President of the United States."
    },
    "joe_biden": {
        "name": "Joe Biden",
        "party": "Democratic",
        "bio": "46th President of the United States."
    },
    # Add more personae as needed
}

# Persona management
class PersonaManager:
    """Manages political personae."""
    
    def __init__(self):
        self.personas = personas
        self.active_persona_id = "donald_trump"  # Default
    
    def set_active_persona(self, persona_id):
        """Set the active persona."""
        if persona_id not in self.personas:
            raise ValueError(f"Unknown persona: {persona_id}")
        self.active_persona_id = persona_id
    
    def get_active_persona(self):
        """Get the active persona."""
        return self.personas[self.active_persona_id]

# Instantiate the persona manager
persona_manager = PersonaManager()

# Convenience functions
def get_active_persona_id():
    """Get the active persona ID."""
    return persona_manager.active_persona_id

def get_persona_name(persona_id):
    """Get the name of a persona."""
    return persona_manager.personas[persona_id]["name"]

def get_persona_party(persona_id):
    """Get the party of a persona."""
    return persona_manager.personas[persona_id]["party"]

def select_persona(persona_id):
    """Select a persona as active."""
    persona_manager.set_active_persona(persona_id)

# Don't import graph at init level to avoid circular imports
# Instead, provide this function that can be imported where needed
def get_graph():
    """Import and return the graph to avoid circular imports."""
    from .graph import graph
    return graph

# Import run_conversation at the end to avoid circular imports
from .graph import run_conversation

# Import other components for convenience
from .config import get_config, get_model_for_task

# At the end of the file, add:
from .graph import run_conversation, run_conversation_with_tracing  # Add the tracing function