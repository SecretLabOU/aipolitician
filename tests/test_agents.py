"""Tests for PoliticianAI agents."""

import pytest
from unittest.mock import MagicMock, patch

from src.agents.base import BaseAgent
from src.agents.context_agent import ContextAgent
from src.agents.response_agent import ResponseAgent
from src.agents.sentiment_agent import SentimentAgent
from src.agents.workflow_manager import WorkflowManager

# Test data
TEST_TEXT = "What is your stance on healthcare reform?"
TEST_CONTEXT = {
    "session_id": "test-session",
    "chat_history": [
        {
            "user_input": "Hello",
            "system_response": "Hi, how can I help you?",
            "sentiment": 0.8,
            "topics": ["General"]
        }
    ]
}

class TestBaseAgent:
    """Tests for BaseAgent class."""
    
    class ConcreteAgent(BaseAgent):
        """Concrete implementation for testing."""
        def process(self, input_data, context=None, db=None, **kwargs):
            return {"result": input_data}
    
    def test_validate_input(self):
        """Test input validation."""
        agent = self.ConcreteAgent()
        
        # Valid input
        assert agent.validate_input("test") is True
        
        # Invalid input
        assert agent.validate_input("") is False
        assert agent.validate_input(None) is False
        assert agent.validate_input(123) is False
    
    def test_preprocess(self):
        """Test preprocessing."""
        agent = self.ConcreteAgent()
        result = agent.preprocess("  test  ")
        assert result == "test"
    
    def test_postprocess(self):
        """Test postprocessing."""
        agent = self.ConcreteAgent()
        data = {"key": "value"}
        result = agent.postprocess(data)
        assert result == data
    
    def test_call(self):
        """Test __call__ method."""
        agent = self.ConcreteAgent()
        result = agent("test")
        assert result == {"success": True, "result": {"result": "test"}}

class TestSentimentAgent:
    """Tests for SentimentAgent class."""
    
    @pytest.fixture
    def agent(self):
        """Create SentimentAgent instance."""
        with patch("transformers.pipeline") as mock_pipeline:
            mock_pipeline.return_value = lambda x: [
                {"label": "POSITIVE", "score": 0.9}
            ]
            agent = SentimentAgent()
            return agent
    
    def test_process(self, agent):
        """Test sentiment analysis."""
        result = agent.process(TEST_TEXT)
        assert "score" in result
        assert "label" in result
        assert "confidence" in result
        assert result["score"] > 0
        assert result["label"] == "POSITIVE"

class TestContextAgent:
    """Tests for ContextAgent class."""
    
    @pytest.fixture
    def agent(self):
        """Create ContextAgent instance."""
        with patch("transformers.pipeline") as mock_pipeline:
            mock_pipeline.return_value = lambda text, candidate_labels, multi_label: {
                "labels": ["Healthcare"],
                "scores": [0.9]
            }
            agent = ContextAgent()
            return agent
    
    def test_process(self, agent):
        """Test context extraction."""
        result = agent.process(TEST_TEXT)
        assert "topics" in result
        assert "topic_ids" in result
        assert "scores" in result
        assert len(result["topics"]) > 0
        assert "Healthcare" in result["topics"]

class TestResponseAgent:
    """Tests for ResponseAgent class."""
    
    @pytest.fixture
    def agent(self):
        """Create ResponseAgent instance."""
        with patch("transformers.pipeline") as mock_pipeline:
            mock_pipeline.return_value = lambda *args, **kwargs: [
                {"generated_text": "Here's my stance on healthcare..."}
            ]
            agent = ResponseAgent()
            return agent
    
    def test_process(self, agent):
        """Test response generation."""
        result = agent.process(TEST_TEXT, context=TEST_CONTEXT)
        assert "response" in result
        assert "sources" in result
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0

class TestWorkflowManager:
    """Tests for WorkflowManager class."""
    
    @pytest.fixture
    def manager(self):
        """Create WorkflowManager instance with mocked agents."""
        manager = WorkflowManager()
        
        # Mock sentiment agent
        manager.sentiment_agent = MagicMock()
        manager.sentiment_agent.return_value = {
            "success": True,
            "result": {"score": 0.8}
        }
        
        # Mock context agent
        manager.context_agent = MagicMock()
        manager.context_agent.return_value = {
            "success": True,
            "result": {
                "topics": ["Healthcare"],
                "topic_ids": [1],
                "scores": [0.9]
            }
        }
        
        # Mock response agent
        manager.response_agent = MagicMock()
        manager.response_agent.return_value = {
            "success": True,
            "result": {
                "response": "Here's my stance on healthcare...",
                "sources": []
            }
        }
        
        return manager
    
    def test_process_message(self, manager):
        """Test message processing workflow."""
        result = manager.process_message(TEST_TEXT)
        
        assert "response" in result
        assert "sentiment" in result
        assert "topics" in result
        assert "topic_ids" in result
        assert "session_id" in result
        
        assert isinstance(result["response"], str)
        assert isinstance(result["sentiment"], float)
        assert isinstance(result["topics"], list)
        assert isinstance(result["topic_ids"], list)
        assert isinstance(result["session_id"], str)
        
        # Verify agent calls
        manager.sentiment_agent.assert_called_once()
        manager.context_agent.assert_called_once()
        manager.response_agent.assert_called_once()
