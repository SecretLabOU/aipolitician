"""Workflow manager for orchestrating agent interactions."""

import logging
from typing import Any, Dict

from src.agents.dialogue_generation_agent import DialogueGenerationAgent
from src.utils import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

class WorkflowManager:
    """Manages the workflow between different agents."""
    
    def __init__(self):
        """Initialize workflow manager."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize dialogue generation agent
        self.response_agent = DialogueGenerationAgent()
    
    def process_message(
        self,
        message: str,
        agent: str
    ) -> Dict[str, Any]:
        """
        Process user message through the agent workflow.
        
        Args:
            message: User input message
            agent: Type of agent to use ("trump" or "biden")
            
        Returns:
            Dictionary containing:
                - response: Generated response text
        """
        try:
            # Generate response
            response_context = {"agent": agent}
            response_result = self.response_agent(message, context=response_context)
            
            if not response_result["success"]:
                raise Exception(f"Response generation failed: {response_result['error']}")
            
            return {
                "response": response_result["result"]["response"]
            }
            
        except Exception as e:
            self.logger.error(f"Error in workflow: {str(e)}")
            raise
