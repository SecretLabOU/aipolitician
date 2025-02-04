"""Context extraction agent for PoliticianAI."""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from sqlalchemy.orm import Session
from transformers import pipeline

from src.agents.base import BaseAgent
from src.config import CONTEXT_MODEL, DEVICE, MODEL_PRECISION, POLITICAL_TOPICS
from src.database.models import Topic
from src.utils import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

class ContextAgent(BaseAgent):
    """Agent for extracting context and topics from text."""
    
    def __init__(self):
        """Initialize context agent."""
        super().__init__()
        
        # Initialize zero-shot classification pipeline
        self.pipeline = pipeline(
            task="zero-shot-classification",
            model=CONTEXT_MODEL,
            device=DEVICE,
            torch_dtype=MODEL_PRECISION
        )
        
        # Cache topic mappings
        self.topic_cache: Dict[str, int] = {}
    
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate input text.
        
        Args:
            input_data: Input text to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(input_data, str):
            return False
        if not input_data.strip():
            return False
        return True
    
    def preprocess(
        self,
        input_data: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Preprocess input text.
        
        Args:
            input_data: Input text to preprocess
            context: Optional context dictionary
            
        Returns:
            Preprocessed text
        """
        # Clean and normalize text
        text = input_data.strip()
        
        # Truncate if too long (model max length)
        if len(text) > 512:
            text = text[:512]
        
        return text
    
    def process(
        self,
        input_data: str,
        context: Optional[Dict[str, Any]] = None,
        db: Optional[Session] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Extract context and topics from input text.
        
        Args:
            input_data: Input text to analyze
            context: Optional context dictionary
            db: Optional database session
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary containing:
                - topics: List of identified topics
                - topic_ids: List of topic IDs
                - scores: Topic confidence scores
        """
        try:
            # Get or create topics in database
            topics = self._get_or_create_topics(db) if db else POLITICAL_TOPICS
            
            # Run zero-shot classification
            result = self.pipeline(
                input_data,
                candidate_labels=topics,
                multi_label=True
            )
            
            # Get topics above threshold (0.5)
            identified_topics = []
            topic_scores = []
            for topic, score in zip(result["labels"], result["scores"]):
                if score > 0.5:
                    identified_topics.append(topic)
                    topic_scores.append(score)
            
            # Get topic IDs if database is available
            topic_ids = []
            if db and self.topic_cache:
                topic_ids = [
                    self.topic_cache[topic]
                    for topic in identified_topics
                    if topic in self.topic_cache
                ]
            
            return {
                "topics": identified_topics,
                "topic_ids": topic_ids,
                "scores": topic_scores
            }
            
        except Exception as e:
            self.logger.error(f"Error in context extraction: {str(e)}")
            raise
    
    def postprocess(
        self,
        output_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Postprocess context extraction results.
        
        Args:
            output_data: Context extraction results
            context: Optional context dictionary
            
        Returns:
            Postprocessed results
        """
        # Round scores to 3 decimal places
        output_data["scores"] = [
            round(score, 3) for score in output_data["scores"]
        ]
        
        return output_data
    
    def _get_or_create_topics(self, db: Session) -> List[str]:
        """
        Get existing topics or create them in database.
        
        Args:
            db: Database session
            
        Returns:
            List of topic names
        """
        try:
            # Clear cache if empty
            if not self.topic_cache:
                # Get existing topics
                existing_topics = db.query(Topic).all()
                
                # Create missing topics
                for topic in POLITICAL_TOPICS:
                    topic_obj = next(
                        (t for t in existing_topics if t.name == topic),
                        None
                    )
                    if not topic_obj:
                        topic_obj = Topic(name=topic)
                        db.add(topic_obj)
                        db.flush()  # Get ID without committing
                    
                    self.topic_cache[topic] = topic_obj.id
                
                db.commit()
            
            return list(self.topic_cache.keys())
            
        except Exception as e:
            self.logger.error(f"Error getting/creating topics: {str(e)}")
            db.rollback()
            return POLITICAL_TOPICS
