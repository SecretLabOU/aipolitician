"""Politician database and persona management.

Simple database for managing politician personas and their attributes.
Loads data from personas.json and provides easy access to the politician's style and views.
"""
import json
import os
import logging
from typing import Dict, List, Optional, Any

# Configure logging
logger = logging.getLogger(__name__)

class PersonaManager:
    """Manages politician personas and conversation memory."""
    
    def __init__(self):
        """Initialize the persona manager."""
        self.personas = {}
        self.active_persona_id = None
        self.conversation_memory = {}
        self.load_personas()
    
    def load_personas(self) -> None:
        """Load personas from the JSON file."""
        file_path = os.path.join(os.path.dirname(__file__), "personas.json")
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                for persona in data.get("personas", []):
                    self.personas[persona["id"]] = persona
            
            # Set default persona if needed
            if self.personas and not self.active_persona_id:
                self.active_persona_id = list(self.personas.keys())[0]
            
            logger.info(f"Loaded {len(self.personas)} personas from {file_path}")
        except Exception as e:
            logger.error(f"Error loading personas: {str(e)}")
    
    def select_persona(self, persona_id: str) -> bool:
        """Select a specific persona by ID."""
        if persona_id in self.personas:
            self.active_persona_id = persona_id
            logger.info(f"Selected persona: {persona_id}")
            return True
        logger.warning(f"Persona not found: {persona_id}")
        return False
    
    def get_active_persona(self) -> Dict[str, Any]:
        """Get the currently active persona."""
        if not self.active_persona_id or self.active_persona_id not in self.personas:
            raise ValueError("No active persona selected")
        return self.personas[self.active_persona_id]
    
    def store_memory(self, key: str, value: Any) -> None:
        """Store information in conversation memory."""
        if not self.active_persona_id:
            return
        
        if self.active_persona_id not in self.conversation_memory:
            self.conversation_memory[self.active_persona_id] = {}
        
        self.conversation_memory[self.active_persona_id][key] = value
    
    def get_all_memory(self) -> Dict[str, Any]:
        """Get all memory for the active persona."""
        if not self.active_persona_id:
            return {}
        
        return self.conversation_memory.get(self.active_persona_id, {})


# Initialize the persona manager
persona_manager = PersonaManager()

# Database query functions
def query_voting_db(query: str) -> str:
    """Query voting database for the active persona."""
    try:
        persona = persona_manager.get_active_persona()
        policy_stances = persona.get("policy_stances", {})
        
        response = []
        for policy_area, details in policy_stances.items():
            if policy_area.lower() in query.lower():
                response.append(f"{policy_area.capitalize()}: {details['position']}")
                response.append(f"Key proposals: {', '.join(details.get('key_proposals', ['']))}")
        
        return "\n".join(response) if response else f"No voting records found for {persona['name']} on this topic."
    except Exception:
        return "No voting records found."

def query_bio_db(query: str) -> str:
    """Query biography database for the active persona."""
    try:
        persona = persona_manager.get_active_persona()
        bio = persona.get("biography", {})
        
        return f"Born: {bio.get('birthDate', '')}\nPlace: {bio.get('birthPlace', '')}\nEducation: {bio.get('education', '')}\nCareer: {', '.join(bio.get('career', []))}"
    except Exception:
        return "No biographical information available."

def query_social_db(query: str) -> str:
    """Query social media database for the active persona."""
    try:
        persona = persona_manager.get_active_persona()
        speech = persona.get("speech_patterns", {})
        media = persona.get("media_portrayal", {})
        
        return f"Communication style: {speech.get('sentence_structure', '')}\nCommon phrases: {', '.join(speech.get('catchphrases', [])[:3])}\nPreferred platforms: {', '.join(media.get('positive_coverage_sources', []))}"
    except Exception:
        return "No social media data available."

def query_policy_db(query: str) -> str:
    """Query policy database for the active persona."""
    try:
        persona = persona_manager.get_active_persona()
        policy_stances = persona.get("policy_stances", {})
        
        # Check for specific policy area
        for policy_area, details in policy_stances.items():
            if policy_area.lower() in query.lower():
                talking_points = '\n'.join([f"• {point}" for point in details.get('talking_points', [])])
                return f"Position on {policy_area}: {details['position']}\n\nTalking points:\n{talking_points}"
        
        # General policy summary
        positions = '\n'.join([f"• {area}: {details['position']}" for area, details in policy_stances.items()])
        return f"{persona['name']}'s key positions:\n{positions}"
    except Exception:
        return "No policy information available."

def query_persona_db(query: str) -> str:
    """Query persona database for rhetorical style and speech patterns."""
    try:
        persona = persona_manager.get_active_persona()
        speech = persona.get("speech_patterns", {})
        rhetoric = persona.get("rhetorical_style", {})
        
        return f"Vocabulary: {', '.join(speech.get('vocabulary', [])[:5])}\nDebate tactics: {', '.join(rhetoric.get('debate_tactics', [])[:3])}\nPersuasion: {', '.join(rhetoric.get('persuasion_techniques', [])[:3])}"
    except Exception:
        return "No persona style information available."

def query_chat_memory_db(query: str) -> str:
    """Query chat memory database for conversation context."""
    memory = persona_manager.get_all_memory()
    if not memory:
        return "No previous conversation context."
    
    return "\n".join([f"{topic}: {details}" for topic, details in memory.items()])

def query_factual_kb(query: str) -> str:
    """Query factual knowledge base for verified information."""
    try:
        persona = persona_manager.get_active_persona()
        return f"Name: {persona.get('name')}\nRole: {persona.get('role')}\nParty: {persona.get('party')}"
    except Exception:
        return "No factual information available."

# Utility functions
def select_persona(persona_id: str) -> str:
    """Select a specific persona by ID."""
    success = persona_manager.select_persona(persona_id)
    if success:
        persona = persona_manager.get_active_persona()
        return f"Selected: {persona['name']}"
    else:
        return f"Persona not found: {persona_id}"

def store_conversation_memory(topic: str, details: str) -> str:
    """Store information in conversation memory."""
    persona_manager.store_memory(topic, details)
    return f"Stored memory: {topic}"

def get_available_personas() -> str:
    """Get a list of available personas."""
    personas = [f"{pid}: {persona['name']}" for pid, persona in persona_manager.personas.items()]
    return "\n".join(personas) if personas else "No personas available."

# Registry of all database functions
DB_REGISTRY = {
    "voting": query_voting_db,
    "bio": query_bio_db,
    "social": query_social_db,
    "policy": query_policy_db,
    "persona": query_persona_db,
    "chat_memory": query_chat_memory_db,
    "factual_kb": query_factual_kb,
    "select_persona": select_persona,
    "store_memory": store_conversation_memory,
    "list_personas": get_available_personas,
}
