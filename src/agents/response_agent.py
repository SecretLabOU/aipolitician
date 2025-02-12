"""Response generation agent for PoliticianAI."""

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session
from transformers import pipeline

from src.agents.base import BaseAgent
from src.config import DEVICE, MODEL_PRECISION, RESPONSE_MODEL
from src.database.models import ChatHistory, Politician, Statement, Topic
from src.utils import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

class ResponseAgent(BaseAgent):
    """Agent for generating contextual responses."""
    
    def __init__(self):
        """Initialize response agent."""
        super().__init__()
        
        # Initialize text generation pipeline with GPT-2 model
        self.pipeline = pipeline(
            task="text-generation",
            model=RESPONSE_MODEL,
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
            Preprocessed text with context
        """
        # Clean and normalize text
        text = input_data.strip()
        
        # Add context if available
        if context:
            # Add agent type for response styling
            agent = context.get("agent", "")
            if agent:
                text = f"Style: {agent.upper()}\n{text}"
            
            # Add identified topics
            topics = context.get("topics", [])
            if topics:
                text = f"Topics: {', '.join(topics)}\nQuestion: {text}"
            
            # Add sentiment context
            sentiment = context.get("sentiment")
            if sentiment is not None:
                sentiment_label = "positive" if sentiment > 0 else "negative"
                text = f"Tone: {sentiment_label}\n{text}"
            
            # Add chat history context
            history = context.get("chat_history", [])
            if history:
                history_text = "\n".join([
                    f"User: {entry['user_input']}\nSystem: {entry['system_response']}"
                    for entry in history[-2:]  # Last 2 exchanges
                ])
                text = f"Previous conversation:\n{history_text}\n\nCurrent: {text}"
        
        return text
    
    def process(
        self,
        input_data: str,
        context: Optional[Dict[str, Any]] = None,
        db: Optional[Session] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate response based on input and context.
        
        Args:
            input_data: Input text to respond to
            context: Optional context dictionary
            db: Optional database session
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary containing:
                - response: Generated response text
                - sources: List of source statements used
        """
        try:
            sources = []
            
            # Get relevant statements from database if available
            if db and context and "topic_ids" in context:
                sources = self._get_relevant_statements(
                    input_data,
                    context["topic_ids"],
                    db
                )
            
            # Generate response
            generation_input = input_data
            if sources:
                # Add source information to input
                source_text = "\n".join([
                    f"- {source.content}"
                    for source in sources[:3]  # Use top 3 most relevant sources
                ])
                generation_input = f"Context:\n{source_text}\n\nQuestion: {input_data}"
            
            # Generate response with agent-specific styling
            agent_style = ""
            if context and "agent" in context:
                if context["agent"] == "trump":
                    agent_style = "Respond in Donald Trump's speaking style - use simple words, strong opinions, and phrases like 'believe me', 'tremendous', 'huge'. Be assertive and use exclamation points."
                elif context["agent"] == "biden":
                    agent_style = "Respond in Joe Biden's speaking style - use folksy language, personal anecdotes, phrases like 'folks', 'look', 'here's the deal'. Be empathetic and measured."
            
            generation_input = f"{agent_style}\n{generation_input}" if agent_style else generation_input
            
            # Generate response with GPT-2 model
            response = self.pipeline(
                generation_input,
                max_length=200,  # Longer responses for more natural conversation
                min_length=50,   # Ensure substantive responses
                num_return_sequences=1,
                temperature=0.9,  # Higher temperature for more creative responses
                top_k=50,        # Maintain diversity while avoiding nonsense
                top_p=0.95,      # Nucleus sampling for natural text
                do_sample=True,  # Enable sampling for more varied responses
                pad_token_id=self.pipeline.tokenizer.eos_token_id
            )[0]["generated_text"]
            
            # Extract the response after the input prompt
            response = response[len(generation_input):].strip()
            
            return {
                "response": response,
                "sources": [
                    {
                        "content": source.content,
                        "politician": source.politician.name,
                        "topic": source.topic.name
                    }
                    for source in sources
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            raise
    
    def postprocess(
        self,
        output_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Postprocess generated response.
        
        Args:
            output_data: Response generation results
            context: Optional context dictionary
            
        Returns:
            Postprocessed results
        """
        # Clean up response text
        response = output_data["response"].strip()
        
        # Remove any "Question:" or "Context:" prefixes
        response = response.replace("Question:", "").replace("Context:", "").strip()
        
        output_data["response"] = response
        return output_data
    
    def _get_relevant_statements(
        self,
        query: str,
        topic_ids: List[int],
        db: Session,
        limit: int = 5
    ) -> List[Statement]:
        """
        Get relevant statements from database.
        
        Args:
            query: Input text to find relevant statements for
            topic_ids: List of relevant topic IDs
            db: Database session
            limit: Maximum number of statements to return
            
        Returns:
            List of relevant statements
        """
        try:
            # For now, just get recent statements for the topics
            # In a real implementation, this would use semantic search
            statements = (
                db.query(Statement)
                .filter(Statement.topic_id.in_(topic_ids))
                .order_by(Statement.date.desc())
                .limit(limit)
                .all()
            )
            
            return statements
            
        except Exception as e:
            self.logger.error(f"Error getting relevant statements: {str(e)}")
            return []
