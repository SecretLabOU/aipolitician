from typing import Dict, List, Optional
from transformers import pipeline
import torch
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

from .base import BaseAgent
from src.config import CONTEXT_MODEL, POLITICAL_TOPICS

class ContextAgent(BaseAgent):
    """Agent for extracting context and identifying topics in political discourse"""
    
    def __init__(
        self,
        memory: Optional[ConversationBufferMemory] = None,
        verbose: bool = False,
        threshold: float = 0.3
    ):
        super().__init__(
            name="ContextAgent",
            description="Extracts context and identifies topics in political discourse",
            memory=memory,
            verbose=verbose
        )
        
        # Initialize zero-shot classification pipeline
        self.classifier = pipeline(
            "zero-shot-classification",
            model=CONTEXT_MODEL,
            device=0 if torch.cuda.is_available() else -1
        )
        
        self.threshold = threshold
        self.topics = POLITICAL_TOPICS

    async def arun(self, query: str) -> Dict:
        """
        Asynchronously extract context
        
        Args:
            query: Text to analyze
            
        Returns:
            Dict containing context analysis results
        """
        return self.run(query)

    def run(self, query: str) -> Dict:
        """
        Extract context and identify topics
        
        Args:
            query: Text to analyze
            
        Returns:
            Dict containing:
                - main_topic: Primary identified topic
                - confidence: Confidence score for main topic
                - related_topics: List of related topics with scores
                - context: Relevant contextual information
        """
        if not self._validate_input(query):
            return {
                'main_topic': 'general',
                'confidence': 0.0,
                'related_topics': [],
                'context': []
            }

        try:
            # Classify topics
            result = self.classifier(
                query,
                candidate_labels=self.topics,
                multi_label=True
            )

            # Filter topics above threshold
            valid_topics = [
                {'topic': label, 'score': score}
                for label, score in zip(result['labels'], result['scores'])
                if score > self.threshold
            ]

            # Get relevant context
            context = self._get_relevant_context(query)

            response = {
                'main_topic': result['labels'][0],
                'confidence': float(result['scores'][0]),
                'related_topics': valid_topics[1:],  # Exclude main topic
                'context': context
            }

            # Update memory if available
            self._update_memory(query, str(response))

            return response

        except Exception as e:
            if self.verbose:
                print(f"Error in context extraction: {str(e)}")
            return {
                'main_topic': 'general',
                'confidence': 0.0,
                'related_topics': [],
                'context': []
            }

    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract political entities from text
        
        Args:
            text: Input text
            
        Returns:
            List of dictionaries containing entity information
        """
        try:
            # Use zero-shot classification for entity types
            entity_types = ['politician', 'organization', 'location', 'policy', 'event']
            
            result = self.classifier(
                text,
                candidate_labels=entity_types,
                multi_label=True
            )
            
            entities = []
            for label, score in zip(result['labels'], result['scores']):
                if score > self.threshold:
                    entities.append({
                        'type': label,
                        'value': text,  # In a real implementation, you'd want to extract the specific entity text
                        'confidence': float(score)
                    })
            
            return entities
            
        except Exception as e:
            if self.verbose:
                print(f"Error in entity extraction: {str(e)}")
            return []

    def analyze_topic_relationships(self, topics: List[str]) -> List[Dict]:
        """
        Analyze relationships between topics
        
        Args:
            topics: List of topics to analyze
            
        Returns:
            List of dictionaries containing topic relationships
        """
        relationships = []
        try:
            for i, topic1 in enumerate(topics):
                for topic2 in topics[i+1:]:
                    # Get context for both topics
                    context1 = self._get_relevant_context(topic1)
                    context2 = self._get_relevant_context(topic2)
                    
                    # Analyze relationship using zero-shot classification
                    if context1 and context2:
                        combined_context = f"{context1[0]} {context2[0]}"
                        result = self.classifier(
                            combined_context,
                            candidate_labels=['related', 'opposing', 'independent'],
                            multi_label=False
                        )
                        
                        relationships.append({
                            'topic1': topic1,
                            'topic2': topic2,
                            'relationship': result['labels'][0],
                            'confidence': float(result['scores'][0])
                        })
            
            return relationships
            
        except Exception as e:
            if self.verbose:
                print(f"Error in relationship analysis: {str(e)}")
            return relationships
