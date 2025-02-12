"""Workflow manager for orchestrating agent interactions."""

import logging
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from src.agents.base import BaseAgent
from src.agents.dialogue_generation_agent import DialogueGenerationAgent
from src.database.models import ChatHistory
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
        agent: str,
        db: Optional[Session] = None
    ) -> Dict[str, Any]:
        """
        Process user message through the agent workflow.
        
        Args:
            message: User input message
            agent: Type of agent to use ("trump" or "biden")
            db: Optional database session
            
        Returns:
            Dictionary containing:
                - response: Generated response text
                - session_id: Chat session ID
        """
        try:
            # Generate session ID
            session_id = str(uuid.uuid4())
            
            # Get chat history for context
            context = self._get_chat_context(f"{agent}_{session_id}", db) if db else {}
            
            # Generate response
            response_context = {
                "chat_history": context.get("chat_history", []),
                "agent": agent
            }
            response_result = self.response_agent(
                message,
                context=response_context,
                db=db
            )
            if not response_result["success"]:
                raise Exception(f"Response generation failed: {response_result['error']}")
            
            return {
                "response": response_result["result"]["response"],
                "session_id": session_id
            }
            
        except Exception as e:
            self.logger.error(f"Error in workflow: {str(e)}")
            raise
    
    def _get_chat_context(
        self,
        session_id: str,
        db: Session,
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Get chat history context for a session.
        
        Args:
            session_id: Chat session ID
            db: Database session
            limit: Maximum number of history entries to return
            
        Returns:
            Dictionary containing chat context
        """
        try:
            # Get recent chat history
            history = (
                db.query(ChatHistory)
                .filter(ChatHistory.session_id == session_id)
                .order_by(ChatHistory.created_at.desc())
                .limit(limit)
                .all()
            )
            
            # Format history entries
            chat_history = []
            for entry in reversed(history):  # Oldest to newest
                chat_history.append({
                    "user_input": entry.user_input,
                    "system_response": entry.system_response
                })
            
            return {
                "session_id": session_id,
                "chat_history": chat_history
            }
            
        except Exception as e:
            self.logger.error(f"Error getting chat context: {str(e)}")
            return {
                "session_id": session_id,
                "chat_history": []
            }
