"""PoliticianAI package."""

from src.agents.dialogue_generation_agent import DialogueGenerationAgent
from src.agents.workflow_manager import WorkflowManager
from src.config import *
from src.database.models import ChatHistory

__version__ = "1.0.0"

__all__ = [
    'DialogueGenerationAgent',
    'WorkflowManager',
    'ChatHistory'
]
