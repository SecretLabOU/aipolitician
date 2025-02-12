"""PoliticianAI package."""

__version__ = "1.0.0"

from src.agents.dialogue_generation_agent import DialogueGenerationAgent
from src.agents.workflow_manager import WorkflowManager

__all__ = [
    'DialogueGenerationAgent',
    'WorkflowManager'
]
