"""Political agent graph.

This module implements a LangGraph for simulating politicians.
"""

import json
import os
from pathlib import Path

# Import config module for persona selection
from political_agent_graph.config import select_persona as config_select_persona

# Set up the persona manager
class PersonaManager:
    """Manages loading and accessing politician personas."""
    
    def __init__(self):
        self.personas = {}
        self.active_persona = None
        self._load_personas()
    
    def _load_personas(self):
        """Load all personas from the personas.json file."""
        current_dir = Path(__file__).parent
        try:
            with open(current_dir / "personas.json", "r") as f:
                data = json.load(f)
                for persona in data.get("personas", []):
                    if "id" in persona:
                        self.personas[persona["id"]] = persona
        except FileNotFoundError:
            print("Warning: personas.json not found")
        except json.JSONDecodeError:
            print("Warning: personas.json has invalid format")
    
    def get_active_persona(self):
        """Get the currently active persona."""
        if self.active_persona is None:
            # Default to first persona if none is active
            if self.personas:
                first_id = list(self.personas.keys())[0]
                self.active_persona = self.personas[first_id]
        return self.active_persona
    
    def set_active_persona(self, persona_id):
        """Set the active persona by ID."""
        if persona_id in self.personas:
            self.active_persona = self.personas[persona_id]
            # Also update the config module's active persona
            config_select_persona(persona_id)
            return True
        return False

# Initialize persona manager
persona_manager = PersonaManager()

# Initialize the persona models
from political_agent_graph.persona_models import initialize_persona_models
initialize_persona_models()

# Import the graph functions
from political_agent_graph.graph import run_conversation

# Public API
def select_persona(persona_id):
    """Select a persona to use for conversation."""
    return persona_manager.set_active_persona(persona_id)