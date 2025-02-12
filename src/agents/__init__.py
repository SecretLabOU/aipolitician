"""Agent modules for PoliticianAI."""

# Import base first to avoid circular imports
from src.agents.base import BaseAgent

# Then import specific agents
from src.agents.dialogue_generation_agent import DialogueGenerationAgent
from src.agents.workflow_manager import WorkflowManager

__all__ = [
    'BaseAgent',
    'DialogueGenerationAgent',
    'WorkflowManager'
]
