"""Workflow manager for orchestrating agent interactions."""

import logging
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from src.agents.base import BaseAgent
from src.agents.context_agent import ContextAgent
from src.agents.response_agent import ResponseAgent
from src.agents.sentiment_agent import SentimentAgent
from src.database.models import ChatHistory, Topic
from src.utils import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

class WorkflowManager:
    """Manages the workflow between different agents."""
    
    def __init__(self):
        """Initialize workflow manager."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize agents
        self.sentiment_agent = SentimentAgent()
        self.context_agent = ContextAgent()
        self.response_agent = ResponseAgent()
    
    def process_message(
        self,
        message: str,
        session_id: Optional[str] = None,
        db: Optional[Session] = None
    ) -> Dict[str, Any]:
        """
        Process user message through the agent workflow.
        
        Args:
            message: User input message
            session_id: Optional session ID for chat history
            db: Optional database session
            
        Returns:
            Dictionary containing:
                - response: Generated response text
                - sentiment: Sentiment score
                - topics: List of identified topics
                - topic_ids: List of topic IDs
                - session_id: Chat session ID
        """
        try:
            # Generate session ID if not provided
            if not session_id:
                session_id = str(uuid.uuid4())
            
            # Get chat history for context
            context = self._get_chat_context(session_id, db) if db else {}
            
            # Analyze sentiment
            sentiment_result = self.sentiment_agent(
                message,
                context=context,
                db=db
            )
            if not sentiment_result["success"]:
                raise Exception(f"Sentiment analysis failed: {sentiment_result['error']}")
            sentiment_score = sentiment_result["result"]["score"]
            
            # Extract context and topics
            context_result = self.context_agent(
                message,
                context=context,
                db=db
            )
            if not context_result["success"]:
                raise Exception(f"Context extraction failed: {context_result['error']}")
            topics = context_result["result"]["topics"]
            topic_ids = context_result["result"]["topic_ids"]
            
            # Generate response
            response_context = {
                "sentiment": sentiment_score,
                "topics": topics,
                "topic_ids": topic_ids,
                "chat_history": context.get("chat_history", [])
            }
            response_result = self.response_agent(
                message,
                context=response_context,
                db=db
            )
            if not response_result["success"]:
                raise Exception(f"Response generation failed: {response_result['error']}")
            response = response_result["result"]["response"]
            
            return {
                "response": response,
                "sentiment": sentiment_score,
                "topics": topics,
                "topic_ids": topic_ids,
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
                    "system_response": entry.system_response,
                    "sentiment": entry.sentiment_score,
                    "topics": entry.context_topics.split(",") if entry.context_topics else []
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
