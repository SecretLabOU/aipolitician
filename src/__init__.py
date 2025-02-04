"""
PoliticianAI - AI system for political discourse simulation
"""

__version__ = "1.0.0"
__author__ = "PoliticianAI Contributors"
__license__ = "MIT"

from src.agents.base import BaseAgent
from src.agents.sentiment_agent import SentimentAgent
from src.agents.context_agent import ContextAgent
from src.agents.response_agent import ResponseAgent
from src.agents.workflow_manager import WorkflowManager

__all__ = [
    "BaseAgent",
    "SentimentAgent",
    "ContextAgent",
    "ResponseAgent",
    "WorkflowManager",
]
