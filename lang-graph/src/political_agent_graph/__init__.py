"""
Political Agent Graph

A streamlined, GPU-accelerated conversation system with political personas.
Uses LangGraph for structured conversation flow with local LLMs.

This module provides the core functionality for the Political Agent System.
"""

__version__ = "1.0.0"

# Public exports
from .graph import run_conversation
from .persona_manager import (
    get_available_personas,
    set_active_persona,
    get_active_persona,
)
from .state import ConversationState, get_initial_state
from .config import get_model_for_task

# Expose the version
__all__ = [
    "run_conversation",
    "get_available_personas",
    "set_active_persona", 
    "get_active_persona",
    "ConversationState",
    "get_initial_state",
    "get_model_for_task",
    "__version__",
]