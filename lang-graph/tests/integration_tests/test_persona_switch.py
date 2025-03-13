import pytest
from unittest.mock import patch, MagicMock
import json
import asyncio

from political_agent_graph.config import select_persona, get_active_persona, PERSONA_MODEL_MAP
from political_agent_graph.state import get_initial_state
from political_agent_graph.graph import run_conversation
from political_agent_graph.local_models import SimpleModel


@pytest.fixture
def mock_simple_model():
    """Create a mock SimpleModel that returns predefined responses based on persona."""
    mock_model = MagicMock(spec=SimpleModel)
    
    def side_effect_generate(prompt, **kwargs):
        persona = get_active_persona()
        if "trump" in persona.lower():
            return "This is a tremendous response, believe me. Nobody does responses better than me."
        elif "biden" in persona.lower():
            return "Look, here's the deal folks. Let me be clear about this response."
        elif "obama" in persona.lower():
            return "Let me be clear. This is the response we've been waiting for."
        else:
            return "Default response from the model."
    
    mock_model.generate_response.side_effect = side_effect_generate
    return mock_model


@pytest.fixture
def reset_active_persona():
    """Reset the active persona after each test."""
    yield
    select_persona("trump")  # Reset to default persona


@pytest.mark.asyncio
async def test_switch_between_personas(mock_simple_model, reset_active_persona):
    """Test switching between different personas and verify behavior changes."""
    # Setup patches to avoid actual model loading
    with patch("political_agent_graph.local_models.get_model", return_value=mock_simple_model):
        # Test Trump persona
        select_persona("trump")
        state = get_initial_state()
        state = await run_conversation("Give me a response", state)
        
        # Verify Trump-like response in history
        assert "tremendous" in state.history[-1]["content"].lower()
        
        # Switch to Biden and test
        select_persona("biden")
        state = await run_conversation("Give me a response", state)
        
        # Verify Biden-like response in history
        assert "folks" in state.history[-1]["content"].lower()
        
        # Switch to Obama and test
        select_persona("obama")
        state = await run_conversation("Give me a response", state)
        
        # Verify Obama-like response in history
        assert "let me be clear" in state.history[-1]["content"].lower()


@pytest.mark.asyncio
async def test_persona_context_loading(mock_simple_model, reset_active_persona):
    """Test that persona context is properly loaded and applied to responses."""
    with patch("political_agent_graph.local_models.get_model", return_value=mock_simple_model):
        # Test with different personas and verify the context is reflected in responses
        personas = ["trump", "biden", "obama"]
        
        for persona in personas:
            select_persona(persona)
            assert get_active_persona() == persona
            
            # Verify the model mapping is correctly updated
            assert persona.lower() in json.dumps(PERSONA_MODEL_MAP).lower()
            
            # Run a conversation and check if response matches persona style
            state = get_initial_state()
            state = await run_conversation("What's your opinion?", state)
            
            last_response = state.history[-1]["content"].lower()
            
            if persona == "trump":
                assert any(term in last_response for term in ["tremendous", "believe me"])
            elif persona == "biden":
                assert any(term in last_response for term in ["folks", "look", "deal"])
            elif persona == "obama":
                assert "clear" in last_response


@pytest.mark.asyncio
async def test_invalid_persona_handling(mock_simple_model, reset_active_persona):
    """Test handling of invalid persona selection."""
    with patch("political_agent_graph.local_models.get_model", return_value=mock_simple_model):
        # Try to set an invalid persona
        with pytest.raises(ValueError, match="Unknown persona"):
            select_persona("nonexistent_persona")
        
        # Verify that the active persona remains unchanged (should still be the default)
        assert get_active_persona() == "trump"  # Assuming Trump is the default
        
        # Run a conversation to verify we're still using the default persona
        state = get_initial_state()
        state = await run_conversation("Give me a response", state)
        
        # Verify Trump-like response in history
        assert "tremendous" in state.history[-1]["content"].lower()


@pytest.mark.asyncio
async def test_persona_switch_mid_conversation(mock_simple_model, reset_active_persona):
    """Test switching personas in the middle of a conversation."""
    with patch("political_agent_graph.local_models.get_model", return_value=mock_simple_model):
        # Start with Trump
        select_persona("trump")
        state = get_initial_state()
        
        # First message with Trump
        state = await run_conversation("Hello, who are you?", state)
        trump_response = state.history[-1]["content"]
        assert "tremendous" in trump_response.lower()
        
        # Switch to Biden mid-conversation
        select_persona("biden")
        
        # Next message should have Biden's style
        state = await run_conversation("Can you say that differently?", state)
        biden_response = state.history[-1]["content"]
        assert "folks" in biden_response.lower() or "deal" in biden_response.lower()
        
        # Verify the conversation history contains both personas' responses
        assert "tremendous" in state.history[-3]["content"].lower()  # Trump's response
        assert "folks" in state.history[-1]["content"].lower() or "deal" in biden_response.lower()  # Biden's response


@pytest.mark.asyncio
async def test_persona_config_issues(reset_active_persona):
    """Test behavior when configuration issues occur with personas."""
    
    # Test with a mock that simulates a config issue
    broken_model = MagicMock()
    broken_model.generate_response.side_effect = Exception("Model configuration error")
    
    with patch("political_agent_graph.local_models.get_model", return_value=broken_model):
        select_persona("trump")
        state = get_initial_state()
        
        # The run_conversation should handle the exception gracefully
        # and return an error message in the state
        state = await run_conversation("Hello there", state)
        
        # Verify that an error message is included in the response
        assert "error" in state.history[-1]["content"].lower() or "failed" in state.history[-1]["content"].lower()


@pytest.mark.asyncio
async def test_persona_affects_graph_nodes(mock_simple_model, reset_active_persona):
    """Test that persona switching properly affects different graph nodes."""
    
    # Create mock functions for specific graph nodes to verify they receive correct persona context
    analyze_sentiment_mock = MagicMock(return_value="mocked_sentiment")
    determine_topic_mock = MagicMock(return_value="mocked_topic")
    
    with patch("political_agent_graph.local_models.get_model", return_value=mock_simple_model), \
         patch("political_agent_graph.graph.analyze_sentiment", analyze_sentiment_mock), \
         patch("political_agent_graph.graph.determine_topic", determine_topic_mock):
        
        # Test with Trump persona
        select_persona("trump")
        state = get_initial_state()
        await run_conversation("What do you think about the economy?", state)
        
        # Verify the persona was used in the node calls
        assert analyze_sentiment_mock.called
        assert "trump" in str(analyze_sentiment_mock.call_args).lower()
        
        # Reset mocks
        analyze_sentiment_mock.reset_mock()
        determine_topic_mock.reset_mock()
        
        # Switch to Biden and test again
        select_persona("biden")
        await run_conversation("What do you think about healthcare?", state)
        
        # Verify the node calls reflect the new persona
        assert analyze_sentiment_mock.called
        assert "biden" in str(analyze_sentiment_mock.call_args).lower()

