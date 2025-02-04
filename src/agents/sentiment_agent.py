from typing import Dict, Optional
from transformers import pipeline
import torch
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.memory import ConversationBufferMemory

from .base import BaseAgent
from src.config import SENTIMENT_MODEL

class SentimentAgent(BaseAgent):
    """Agent for analyzing sentiment in political discourse"""
    
    def __init__(
        self,
        memory: Optional[ConversationBufferMemory] = None,
        verbose: bool = False
    ):
        super().__init__(
            name="SentimentAgent",
            description="Analyzes sentiment and emotional tone in political discourse",
            memory=memory,
            verbose=verbose
        )
        
        # Initialize sentiment pipeline
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=SENTIMENT_MODEL,
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Sentiment score mapping
        self.sentiment_mapping = {
            'POSITIVE': {'score_modifier': 1.0, 'tone': 'positive'},
            'NEGATIVE': {'score_modifier': -1.0, 'tone': 'negative'},
            'NEUTRAL': {'score_modifier': 0.0, 'tone': 'neutral'}
        }

    async def arun(self, query: str) -> Dict:
        """
        Asynchronously analyze sentiment
        
        Args:
            query: Text to analyze
            
        Returns:
            Dict containing sentiment analysis results
        """
        return self.run(query)

    def run(self, query: str) -> Dict:
        """
        Analyze sentiment in the input text
        
        Args:
            query: Text to analyze
            
        Returns:
            Dict containing:
                - sentiment: Overall sentiment (positive/negative/neutral)
                - confidence: Confidence score
                - tone: Mapped tone category
                - intensity: Normalized intensity score
        """
        if not self._validate_input(query):
            return {
                'sentiment': 'NEUTRAL',
                'confidence': 0.0,
                'tone': 'neutral',
                'intensity': 0.0
            }

        try:
            # Get raw sentiment analysis
            result = self.sentiment_pipeline(query)[0]
            label = result['label']
            score = float(result['score'])

            # Map to our sentiment categories
            sentiment_info = self.sentiment_mapping.get(
                label,
                self.sentiment_mapping['NEUTRAL']
            )

            # Calculate intensity
            intensity = score * sentiment_info['score_modifier']

            response = {
                'sentiment': label,
                'confidence': score,
                'tone': sentiment_info['tone'],
                'intensity': intensity
            }

            # Update memory if available
            self._update_memory(query, str(response))

            return response

        except Exception as e:
            if self.verbose:
                print(f"Error in sentiment analysis: {str(e)}")
            return {
                'sentiment': 'NEUTRAL',
                'confidence': 0.0,
                'tone': 'neutral',
                'intensity': 0.0
            }

    def _analyze_emotional_intensity(self, text: str) -> float:
        """
        Analyze the emotional intensity of the text
        
        Args:
            text: Input text
            
        Returns:
            float: Normalized intensity score (-1 to 1)
        """
        try:
            # Get relevant emotional context
            context = self._get_relevant_context(text, k=1)
            
            # If we have context, compare with it
            if context:
                context_sentiment = self.sentiment_pipeline(context[0])[0]
                context_score = float(context_sentiment['score'])
                
                # Compare current sentiment with context
                current_sentiment = self.sentiment_pipeline(text)[0]
                current_score = float(current_sentiment['score'])
                
                # Return difference in intensity
                return current_score - context_score
            
            # If no context, just return current sentiment intensity
            result = self.sentiment_pipeline(text)[0]
            return float(result['score']) * (1 if result['label'] == 'POSITIVE' else -1)
            
        except Exception as e:
            if self.verbose:
                print(f"Error in intensity analysis: {str(e)}")
            return 0.0
