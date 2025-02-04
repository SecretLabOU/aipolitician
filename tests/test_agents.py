import pytest
from unittest.mock import Mock, patch
from sqlalchemy.orm import Session

from src.agents.sentiment_agent import SentimentAgent
from src.agents.context_agent import ContextAgent
from src.agents.response_agent import ResponseAgent
from src.agents.workflow_manager import WorkflowManager

@pytest.fixture
def mock_db_session():
    """Create a mock database session"""
    return Mock(spec=Session)

@pytest.fixture
def sentiment_agent():
    """Create a SentimentAgent instance"""
    return SentimentAgent(verbose=True)

@pytest.fixture
def context_agent():
    """Create a ContextAgent instance"""
    return ContextAgent(verbose=True)

@pytest.fixture
def response_agent(mock_db_session):
    """Create a ResponseAgent instance"""
    return ResponseAgent(db_session=mock_db_session, verbose=True)

@pytest.fixture
def workflow_manager(mock_db_session):
    """Create a WorkflowManager instance"""
    return WorkflowManager(db_session=mock_db_session, verbose=True)

class TestSentimentAgent:
    """Test suite for SentimentAgent"""
    
    def test_sentiment_analysis(self, sentiment_agent):
        """Test sentiment analysis on sample text"""
        text = "I strongly support this healthcare policy"
        result = sentiment_agent.run(text)
        
        assert isinstance(result, dict)
        assert 'sentiment' in result
        assert 'confidence' in result
        assert 'tone' in result
        assert isinstance(result['confidence'], float)
        assert 0 <= result['confidence'] <= 1

class TestContextAgent:
    """Test suite for ContextAgent"""
    
    def test_context_extraction(self, context_agent):
        """Test context extraction on sample text"""
        text = "What is the current healthcare policy?"
        result = context_agent.run(text)
        
        assert isinstance(result, dict)
        assert 'main_topic' in result
        assert 'confidence' in result
        assert isinstance(result['confidence'], float)
        assert 0 <= result['confidence'] <= 1

class TestResponseAgent:
    """Test suite for ResponseAgent"""
    
    @patch('src.agents.response_agent.LlamaCpp')
    def test_response_generation(self, mock_llama, response_agent):
        """Test response generation with mocked LLM"""
        # Mock LLM response
        mock_llama.return_value.generate.return_value = "Sample response about healthcare"
        
        # Test input
        query = "What is the healthcare policy?"
        context = {
            'main_topic': 'healthcare',
            'confidence': 0.9,
            'context': []
        }
        sentiment = {
            'tone': 'neutral',
            'confidence': 0.8
        }
        
        result = response_agent.run(query, context, sentiment)
        assert isinstance(result, str)
        assert len(result) > 0

class TestWorkflowManager:
    """Test suite for WorkflowManager"""
    
    def test_process_input(self, workflow_manager):
        """Test complete workflow processing"""
        query = "What is the current position on healthcare reform?"
        result = workflow_manager.process_input(query)
        
        assert isinstance(result, dict)
        assert 'response' in result
        assert 'metadata' in result
        assert isinstance(result['metadata'], dict)

    def test_conversation_memory(self, workflow_manager):
        """Test conversation memory management"""
        # Clear existing memory
        workflow_manager.clear_memory()
        
        # Process a query
        query = "Tell me about healthcare"
        workflow_manager.process_input(query)
        
        # Check conversation history
        history = workflow_manager.get_conversation_history()
        assert len(history) > 0

@pytest.mark.integration
class TestIntegration:
    """Integration tests for the complete system"""
    
    def test_end_to_end_workflow(self, workflow_manager):
        """Test complete end-to-end workflow"""
        queries = [
            "What is the healthcare policy?",
            "How does this compare to previous policies?",
            "What are the main criticisms?"
        ]
        
        for query in queries:
            result = workflow_manager.process_input(query)
            assert isinstance(result, dict)
            assert 'response' in result
            assert 'metadata' in result
            
        # Verify conversation history
        history = workflow_manager.get_conversation_history()
        assert len(history) == len(queries)

if __name__ == '__main__':
    pytest.main([__file__])
