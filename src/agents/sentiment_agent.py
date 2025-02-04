"""Sentiment analysis agent for PoliticianAI."""

import logging
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session
from transformers import pipeline

from src.agents.base import BaseAgent
from src.config import DEVICE, MODEL_PRECISION, SENTIMENT_MODEL
from src.utils import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

class SentimentAgent(BaseAgent):
    """Agent for analyzing sentiment in text."""
    
    def __init__(self):
        """Initialize sentiment agent."""
        super().__init__()
        
        # Initialize sentiment pipeline
        self.pipeline = pipeline(
            task="sentiment-analysis",
            model=SENTIMENT_MODEL,
            device=DEVICE,
            torch_dtype=MODEL_PRECISION
        )
    
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
        Analyze sentiment of input text.
        
        Args:
            input_data: Input text to analyze
            context: Optional context dictionary
            db: Optional database session
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary containing:
                - score: Sentiment score (-1 to 1)
                - label: Sentiment label (POSITIVE/NEGATIVE)
                - confidence: Model confidence score
        """
        try:
            # Run sentiment analysis
            result = self.pipeline(input_data)[0]
            
            # Convert score to -1 to 1 range
            score = result["score"]
            if result["label"] == "NEGATIVE":
                score = -score
            
            return {
                "score": score,
                "label": result["label"],
                "confidence": result["score"]
            }
            
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {str(e)}")
            raise
    
    def postprocess(
        self,
        output_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Postprocess sentiment results.
        
        Args:
            output_data: Sentiment analysis results
            context: Optional context dictionary
            
        Returns:
            Postprocessed results
        """
        # Round scores to 3 decimal places
        output_data["score"] = round(output_data["score"], 3)
        output_data["confidence"] = round(output_data["confidence"], 3)
        
        return output_data
