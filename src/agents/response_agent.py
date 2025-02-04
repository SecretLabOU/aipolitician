from typing import Dict, List, Optional
from datetime import datetime
import json
from sqlalchemy.orm import Session
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp

from .base import BaseAgent
from src.database.models import Politician, PolicyPosition, VotingRecord, Statement
from src.config import MODEL_DIR, RESPONSE_TEMPLATES

class ResponseAgent(BaseAgent):
    """Agent for generating contextual responses in political discourse"""
    
    def __init__(
        self,
        db_session: Session,
        memory: Optional[ConversationBufferMemory] = None,
        verbose: bool = False
    ):
        super().__init__(
            name="ResponseAgent",
            description="Generates contextual responses based on political data",
            memory=memory,
            verbose=verbose
        )
        
        self.db = db_session
        
        # Initialize LLaMA model for response generation
        self.llm = LlamaCpp(
            model_path=str(MODEL_DIR / "llama-2-7b-chat.gguf"),
            temperature=0.7,
            max_tokens=2048,
            top_p=0.95,
            n_ctx=2048,
            verbose=verbose
        )
        
        # Initialize response chain
        self.response_chain = self._create_response_chain()

    def _create_response_chain(self) -> LLMChain:
        """Create the response generation chain"""
        template = """
        Based on the following information, generate a response to the user's question.
        
        Context:
        {context}
        
        Sentiment: {sentiment}
        Topic: {topic}
        
        Policy Position: {policy_position}
        Voting Record: {voting_record}
        
        Generate a response that:
        1. Addresses the user's question directly
        2. Uses appropriate emotional tone
        3. Cites specific facts and sources
        4. Maintains political neutrality
        
        Question: {question}
        
        Response:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=[
                "context",
                "sentiment",
                "topic",
                "policy_position",
                "voting_record",
                "question"
            ]
        )
        
        return LLMChain(llm=self.llm, prompt=prompt)

    async def arun(self, query: str, context: Dict, sentiment: Dict) -> str:
        """
        Asynchronously generate response
        
        Args:
            query: User's question
            context: Context information from ContextAgent
            sentiment: Sentiment information from SentimentAgent
            
        Returns:
            str: Generated response
        """
        return self.run(query, context, sentiment)

    def run(self, query: str, context: Dict, sentiment: Dict) -> str:
        """
        Generate response based on context and sentiment
        
        Args:
            query: User's question
            context: Context information from ContextAgent
            sentiment: Sentiment information from SentimentAgent
            
        Returns:
            str: Generated response
        """
        if not self._validate_input(query):
            return RESPONSE_TEMPLATES["error"]

        try:
            # Get relevant data from database
            policy_data = self._get_policy_data(context['main_topic'])
            voting_data = self._get_voting_data(context['main_topic'])
            
            if not policy_data and not voting_data:
                return RESPONSE_TEMPLATES["not_found"].format(
                    topic=context['main_topic'],
                    politician="the politician"
                )

            # Generate response
            response = self.response_chain.run(
                context=json.dumps(context.get('context', [])),
                sentiment=sentiment['tone'],
                topic=context['main_topic'],
                policy_position=json.dumps(policy_data),
                voting_record=json.dumps(voting_data),
                question=query
            )

            # Update memory if available
            self._update_memory(query, response)

            return response

        except Exception as e:
            if self.verbose:
                print(f"Error in response generation: {str(e)}")
            return RESPONSE_TEMPLATES["error"]

    def _get_policy_data(self, topic: str) -> List[Dict]:
        """Get relevant policy positions from database"""
        try:
            policies = self.db.query(PolicyPosition).filter(
                PolicyPosition.topic == topic
            ).all()
            
            return [
                {
                    'position': policy.position,
                    'source': policy.source,
                    'date': policy.date_updated.isoformat() if policy.date_updated else None
                }
                for policy in policies
            ]
        except Exception as e:
            if self.verbose:
                print(f"Error fetching policy data: {str(e)}")
            return []

    def _get_voting_data(self, topic: str) -> List[Dict]:
        """Get relevant voting records from database"""
        try:
            votes = self.db.query(VotingRecord).filter(
                VotingRecord.topic == topic
            ).all()
            
            return [
                {
                    'bill_name': vote.bill_name,
                    'vote': vote.vote,
                    'date': vote.date.isoformat() if vote.date else None,
                    'source': vote.source
                }
                for vote in votes
            ]
        except Exception as e:
            if self.verbose:
                print(f"Error fetching voting data: {str(e)}")
            return []

    def _adjust_response_tone(self, response: str, sentiment: Dict) -> str:
        """Adjust response tone based on sentiment"""
        try:
            # Use zero-shot classification to analyze current tone
            result = self.classifier(
                response,
                candidate_labels=['formal', 'casual', 'empathetic', 'neutral'],
                multi_label=False
            )
            
            current_tone = result['labels'][0]
            
            # Adjust based on sentiment
            if sentiment['tone'] == 'positive' and current_tone == 'formal':
                # Make more casual and empathetic
                response = self.response_chain.run(
                    original_response=response,
                    target_tone='empathetic',
                    context="Make this response more friendly while maintaining professionalism"
                )
            elif sentiment['tone'] == 'negative' and current_tone == 'casual':
                # Make more formal and neutral
                response = self.response_chain.run(
                    original_response=response,
                    target_tone='formal',
                    context="Make this response more formal and professional"
                )
            
            return response
            
        except Exception as e:
            if self.verbose:
                print(f"Error adjusting response tone: {str(e)}")
            return response
