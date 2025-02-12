"""PoliticianAI package."""

__version__ = "1.0.0"

# Import models first to avoid circular imports
from src.database.models import Base, ChatHistory

# Then import agents
from src.agents.dialogue_generation_agent import DialogueGenerationAgent
from src.agents.workflow_manager import WorkflowManager

__all__ = [
    'DialogueGenerationAgent',
    'WorkflowManager',
    'Base',
    'ChatHistory'
]
