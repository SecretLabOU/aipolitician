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
        
        # Initialize conversational pipeline with BlenderBot
        self.pipeline = pipeline(
            task="conversational",
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
        # Get agent type and prepare persona instruction
        agent_type = context.get("agent", "") if context else ""
        persona_instruction = ""
        if agent_type == "trump":
            persona_instruction = "You are Donald Trump. Respond in your characteristic style - use simple words, strong opinions, and phrases like 'believe me', 'tremendous', 'huge'. Be assertive and use exclamation points."
        elif agent_type == "biden":
            persona_instruction = "You are Joe Biden. Respond in your characteristic style - use folksy language, personal anecdotes, phrases like 'folks', 'look', 'here's the deal'. Be empathetic and measured."
        
        # Format the input with persona
        if persona_instruction:
            input_data = f"{persona_instruction}\n\nHuman: {input_data}\nAssistant:"
        else:
            input_data = f"Human: {input_data}\nAssistant:"
        
        return input_data.strip()
    
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
            
            # Add source information to input if available
            generation_input = input_data
            if sources:
                source_text = "\n".join([
                    f"- {source.content}"
                    for source in sources[:3]  # Use top 3 most relevant sources
                ])
                generation_input = f"{generation_input}\nRelevant context:\n{source_text}"
            
            try:
                # Format conversation for BlenderBot
                conversation = {
                    "past_user_inputs": [],
                    "generated_responses": [],
                    "text": generation_input
                }
                
                # Generate response
                result = self.pipeline(conversation)
                response = result["generated_text"]
                
                # Ensure we have a response
                if not response.strip():
                    response = "I am having trouble generating a response right now."
                
                result = {
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
                return {
                    "success": True,
                    "result": result
                }
            except Exception as e:
                self.logger.error(f"Failed to generate response: {str(e)}")
                return {
                    "success": False,
                    "error": str(e),
                    "result": {
                        "response": "I am having trouble generating a response right now.",
                        "sources": []
                    }
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
        try:
            # Clean up response text
            response = output_data["response"].strip()
            
            # Remove any prefixes and clean up
            response = response.replace("Question:", "").replace("Context:", "").strip()
            response = response.replace("Assistant:", "").strip()
            
            # Ensure we have a non-empty response
            if not response:
                response = "I am having trouble generating a response right now."
            
            output_data["response"] = response
            return output_data
        except Exception as e:
            self.logger.error(f"Error in postprocess: {str(e)}")
            return {
                "response": "I am having trouble generating a response right now.",
                "sources": []
            }
    
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
