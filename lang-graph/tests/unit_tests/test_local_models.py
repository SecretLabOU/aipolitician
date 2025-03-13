import os
import pytest
from unittest.mock import patch, MagicMock

from political_agent_graph.local_models import (
    setup_models,
    get_model,
    get_tokenizer,
    SimpleModel,
    TrumpLLM,
    BidenLLM,
)


@pytest.fixture
def mock_models_setup():
    """Fixture to simulate models being set up"""
    with patch("political_agent_graph.local_models.MODELS", {}) as mock_models:
        with patch("political_agent_graph.local_models.TOKENIZERS", {}) as mock_tokenizers:
            # Simulate models being loaded
            mock_models["trump"] = MagicMock(spec=TrumpLLM)
            mock_models["biden"] = MagicMock(spec=BidenLLM)
            mock_models["simple"] = SimpleModel()
            
            # Simulate tokenizers being loaded
            mock_tokenizers["trump"] = MagicMock()
            mock_tokenizers["biden"] = MagicMock()
            
            yield mock_models, mock_tokenizers


class TestModelLoading:
    """Tests for model loading functionality"""
    
    @patch("political_agent_graph.local_models.os.path.exists")
    @patch("political_agent_graph.local_models.TrumpLLM")
    @patch("political_agent_graph.local_models.BidenLLM")
    @patch("political_agent_graph.local_models.AutoTokenizer")
    def test_setup_models_with_local_models(
        self, mock_tokenizer, mock_biden, mock_trump, mock_exists
    ):
        """Test setup_models when local models exist"""
        # Configure mocks
        mock_exists.return_value = True
        mock_trump_instance = MagicMock()
        mock_biden_instance = MagicMock()
        mock_trump.return_value = mock_trump_instance
        mock_biden.return_value = mock_biden_instance
        
        mock_trump_tokenizer = MagicMock()
        mock_biden_tokenizer = MagicMock()
        mock_tokenizer.from_pretrained.side_effect = [
            mock_trump_tokenizer, 
            mock_biden_tokenizer
        ]
        
        # Call function under test
        with patch("political_agent_graph.local_models.MODELS", {}) as mock_models:
            with patch("political_agent_graph.local_models.TOKENIZERS", {}) as mock_tokenizers:
                setup_models()
                
                # Verify models were created correctly
                assert "trump" in mock_models
                assert "biden" in mock_models
                assert isinstance(mock_models.get("simple"), SimpleModel)
                
                # Verify tokenizers were loaded
                assert "trump" in mock_tokenizers
                assert "biden" in mock_tokenizers
    
    @patch("political_agent_graph.local_models.os.path.exists")
    def test_setup_models_without_local_models(self, mock_exists):
        """Test setup_models when local models don't exist"""
        # Configure mocks
        mock_exists.return_value = False
        
        # Call function under test
        with patch("political_agent_graph.local_models.MODELS", {}) as mock_models:
            with patch("political_agent_graph.local_models.TOKENIZERS", {}) as mock_tokenizers:
                setup_models()
                
                # Only SimpleModel should be available when local models don't exist
                assert "simple" in mock_models
                assert "trump" not in mock_models
                assert "biden" not in mock_models
                assert len(mock_tokenizers) == 0
    
    def test_get_model(self, mock_models_setup):
        """Test get_model function returns the correct model"""
        mock_models, _ = mock_models_setup
        
        # Test retrieving existing models
        assert get_model("trump") == mock_models["trump"]
        assert get_model("biden") == mock_models["biden"]
        assert isinstance(get_model("simple"), SimpleModel)
        
        # Test retrieving non-existent model
        with pytest.raises(KeyError):
            get_model("non_existent_model")
    
    def test_get_tokenizer(self, mock_models_setup):
        """Test get_tokenizer function returns the correct tokenizer"""
        _, mock_tokenizers = mock_models_setup
        
        # Test retrieving existing tokenizers
        assert get_tokenizer("trump") == mock_tokenizers["trump"]
        assert get_tokenizer("biden") == mock_tokenizers["biden"]
        
        # Test retrieving non-existent tokenizer
        with pytest.raises(KeyError):
            get_tokenizer("non_existent_tokenizer")


class TestSimpleModel:
    """Tests for the SimpleModel class"""
    
    def test_simple_model_initialization(self):
        """Test SimpleModel initializes correctly"""
        model = SimpleModel()
        assert model.predefined_responses == {}
    
    def test_generate_response_with_default(self):
        """Test generate_response returns default answer when no pattern is matched"""
        model = SimpleModel()
        response = model.generate_response("What do you think about taxes?")
        assert response.strip() == "I am a simple test model. I don't have a real response to that."
    
    def test_generate_response_with_predefined_response(self):
        """Test generate_response returns predefined response when pattern is matched"""
        model = SimpleModel()
        # Add predefined response
        model.predefined_responses = {
            "taxes": "I believe taxes should be fair.",
            "healthcare": "Healthcare is important for all citizens."
        }
        
        # Test exact match
        response = model.generate_response("taxes")
        assert response == "I believe taxes should be fair."
        
        # Test partial match
        response = model.generate_response("What do you think about healthcare policy?")
        assert response == "Healthcare is important for all citizens."
        
        # Test no match uses default
        response = model.generate_response("What about foreign policy?")
        assert response.strip() == "I am a simple test model. I don't have a real response to that."
    
    def test_add_predefined_response(self):
        """Test adding a predefined response"""
        model = SimpleModel()
        model.add_predefined_response("economy", "The economy is doing well.")
        
        assert "economy" in model.predefined_responses
        assert model.predefined_responses["economy"] == "The economy is doing well."
        
        # Test the predefined response works
        response = model.generate_response("Tell me about the economy")
        assert response == "The economy is doing well."

