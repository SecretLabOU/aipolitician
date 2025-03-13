"""Integration tests for the complete graph execution."""

import pytest
import asyncio
from unittest.mock import patch, MagicMock

from political_agent_graph.state import ConversationState, get_initial_state
from political_agent_graph.graph import (
    analyze_sentiment,
    determine_topic,
    decide_deflection,
    generate_policy_stance,
    format_response,
    graph,
    run_conversation
)
from political_agent_graph import select_persona
from political_agent_graph.config import get_model_for_task


@pytest.fixture
def mock_model():
    """Fixture that creates a mock model that can be configured with responses."""
    mock = MagicMock()
    
    # Store responses so we can configure them in tests
    mock.responses = {}
    
    # Mock the invoke method to return responses based on the prompt
    def mock_invoke(prompt):
        for key, response in mock.responses.items():
            if key in prompt:
                return response
        # Default response if no keyword match
        return "Default mock response"
    
    mock.invoke = mock_invoke
    return mock


@pytest.fixture
def mock_get_model(mock_model):
    """Fixture to patch get_model_for_task to return our mock model."""
    with patch('political_agent_graph.graph.get_model_for_task', return_value=mock_model):
        yield mock_model


@pytest.fixture
def mock_persona_manager():
    """Fixture to patch the persona manager."""
    trump_persona = {
        "id": "donald_trump",
        "name": "Donald Trump",
        "party": "Republican",
        "rhetorical_style": {
            "debate_tactics": ["Counter-attacking critics", "Deflecting criticism by changing subject"],
            "emotional_tone": {"primary": ["confident", "combative"]}
        },
        "speech_patterns": {
            "vocabulary": ["tremendous", "huge", "great", "amazing"],
            "catchphrases": ["Make America Great Again", "Fake News"]
        },
        "policy_stances": {
            "immigration": {
                "position": "Restrictive",
                "key_proposals": ["Border wall with Mexico"],
                "talking_points": ["We need to secure our borders to keep America safe"]
            }
        }
    }
    
    biden_persona = {
        "id": "joe_biden",
        "name": "Joe Biden",
        "party": "Democratic",
        "rhetorical_style": {
            "debate_tactics": ["Drawing on personal experience", "Appealing to empathy"],
            "emotional_tone": {"primary": ["empathetic", "measured"]}
        },
        "speech_patterns": {
            "vocabulary": ["folks", "look", "here's the deal"],
            "catchphrases": ["Build Back Better", "Soul of the Nation"]
        },
        "policy_stances": {
            "immigration": {
                "position": "Pathway to citizenship, humanitarian approach",
                "key_proposals": ["DACA protection", "Border security technology"],
                "talking_points": ["We can secure our borders while treating people with dignity"]
            }
        }
    }
    
    with patch('political_agent_graph.graph.persona_manager') as mock:
        mock.get_active_persona.return_value = trump_persona
        # Allow the mock to switch personas
        def set_persona(persona_id):
            if persona_id == "donald_trump":
                mock.get_active_persona.return_value = trump_persona
            elif persona_id == "joe_biden":
                mock.get_active_persona.return_value = biden_persona
        
        mock.set_active_persona = set_persona
        yield mock


def test_analyze_sentiment(mock_get_model):
    """Test the analyze_sentiment node in isolation."""
    # Configure mock model response
    mock_get_model.responses = {
        "user_input": "positive"
    }
    
    # Create test state
    state = get_initial_state("I love your policies!")
    
    # Run the node
    result = analyze_sentiment(state)
    
    # Assert the sentiment was correctly set in the state
    assert result.topic_sentiment == "positive"


def test_determine_topic(mock_get_model):
    """Test the determine_topic node in isolation."""
    # Configure mock model response
    mock_get_model.responses = {
        "user_input": "immigration"
    }
    
    # Create test state
    state = get_initial_state("What do you think about border security?")
    
    # Run the node
    result = determine_topic(state)
    
    # Assert the topic was correctly set in the state
    assert result.current_topic == "immigration"


def test_decide_deflection_true(mock_get_model, mock_persona_manager):
    """Test the decide_deflection node when it should deflect."""
    # Configure mock model to indicate deflection
    mock_get_model.responses = {
        "deflection": "true\neconomy"
    }
    
    # Create test state
    state = get_initial_state("What's your stance on gun control?")
    state.current_topic = "gun control"
    state.topic_sentiment = "negative"
    
    # Run the node
    result = decide_deflection(state)
    
    # Assert deflection was correctly decided
    assert result.should_deflect == True
    assert result.deflection_topic == "economy"


def test_decide_deflection_false(mock_get_model, mock_persona_manager):
    """Test the decide_deflection node when it should not deflect."""
    # Configure mock model to indicate no deflection
    mock_get_model.responses = {
        "deflection": "false"
    }
    
    # Create test state
    state = get_initial_state("What's your stance on the economy?")
    state.current_topic = "economy"
    state.topic_sentiment = "positive"
    
    # Run the node
    result = decide_deflection(state)
    
    # Assert deflection was correctly decided
    assert result.should_deflect == False
    assert result.deflection_topic is None


def test_generate_policy_stance(mock_get_model, mock_persona_manager):
    """Test the generate_policy_stance node."""
    # Configure mock model to return a policy stance
    mock_get_model.responses = {
        "policy": "I have the best policy on immigration. We need to build a wall and secure our borders."
    }
    
    # Create test state
    state = get_initial_state("What's your immigration policy?")
    state.current_topic = "immigration"
    state.should_deflect = False
    
    # Run the node
    result = generate_policy_stance(state)
    
    # Assert policy stance was set
    assert "best policy on immigration" in result.policy_stance


def test_format_response(mock_get_model, mock_persona_manager):
    """Test the format_response node."""
    # Configure mock model to return a formatted response
    mock_get_model.responses = {
        "response": "Let me tell you, we have a tremendous immigration policy. It's the best policy ever. We're going to build a wall and Mexico will pay for it. Believe me!"
    }
    
    # Create test state
    state = get_initial_state("What's your immigration policy?")
    state.current_topic = "immigration"
    state.should_deflect = False
    state.policy_stance = "Strong borders, wall with Mexico"
    
    # Run the node
    result = format_response(state)
    
    # Assert response was formatted and history was updated
    assert result.final_response is not None
    assert len(result.final_response) > 0
    assert len(result.conversation_history) > 0
    assert "tremendous immigration policy" in result.final_response


@pytest.mark.asyncio
async def test_full_trump_conversation_flow(mock_get_model, mock_persona_manager):
    """Test the complete conversation flow as Trump with a non-deflection path."""
    # Select Trump persona
    select_persona("donald_trump")
    
    # Configure mock model responses for each node
    mock_get_model.responses = {
        "user_input": "positive\nimmigration",  # For sentiment and topic
        "deflection": "false",  # No deflection
        "policy": "We need strong borders and a wall. America first!",  # Policy stance
        "response": "Let me tell you, our immigration system is a disaster. We need strong borders, and we're building a beautiful wall. Nobody builds walls better than me, believe me!"  # Final response
    }
    
    # Run the conversation
    response = await run_conversation("What are your views on immigration?")
    
    # Assert we got the expected response
    assert "beautiful wall" in response
    assert "immigration" in response.lower()


@pytest.mark.asyncio
async def test_full_trump_conversation_with_deflection(mock_get_model, mock_persona_manager):
    """Test the complete conversation flow as Trump with deflection path."""
    # Select Trump persona
    select_persona("donald_trump")
    
    # Configure mock model responses for each node
    mock_get_model.responses = {
        "user_input": "negative\ngun control",  # For sentiment and topic
        "deflection": "true\neconomy",  # Deflect to economy
        "policy": "We have the best economy ever. Record stock market. Tremendous job numbers!",  # Deflected policy stance
        "response": "Look, I know about gun control, but let me tell you about the economy. It's tremendous, the best ever. Jobs are coming back. Stock market at record highs. America is winning again!"  # Deflected response
    }
    
    # Run the conversation
    response = await run_conversation("What's your position on gun control legislation?")
    
    # Assert we got the expected deflected response
    assert "economy" in response.lower()
    assert "tremendous" in response
    assert "winning" in response


@pytest.mark.asyncio
async def test_persona_switching_different_responses(mock_get_model, mock_persona_manager):
    """Test that different personas give different responses to the same question."""
    # Configure mock responses for Trump
    select_persona("donald_trump")
    mock_get_model.responses = {
        "user_input": "neutral\nimmigration",
        "deflection": "false",
        "policy": "Strong borders, build the wall!",
        "response": "We're going to build a great wall and Mexico will pay for it. Immigration is a disaster right now!"
    }
    
    # Get Trump's response
    trump_response = await run_conversation("What are your immigration policies?")
    
    # Now switch to Biden
    select_persona("joe_biden")
    mock_get_model.responses = {
        "user_input": "neutral\nimmigration",
        "deflection": "false",
        "policy": "Humane border policy, pathway to citizenship",
        "response": "Look folks, we need an immigration system that reflects our values. We can secure our borders while treating people with dignity and providing a pathway to citizenship."
    }
    
    # Get Biden's response
    biden_response = await run_conversation("What are your immigration policies?")
    
    # Assert the responses are different
    assert trump_response != biden_response
    assert "wall" in trump_response.lower()
    assert "folks" in biden_response.lower()
    assert "pathway to citizenship" in biden_response.lower()


@pytest.mark.asyncio
async def test_node_execution_sequence(mock_get_model, mock_persona_manager):
    """Test that all nodes in the graph are executed in the correct sequence."""
    # Select persona
    select_persona("donald_trump")
    
    # Track node execution
    executed_nodes = []
    
    # Original functions to wrap
    original_analyze_sentiment = analyze_sentiment
    original_determine_topic = determine_topic
    original_decide_deflection = decide_deflection
    original_generate_policy_stance = generate_policy_stance
    original_format_response = format_response
    
    # Configure mock responses
    mock_get_model.responses = {
        "user_input": "positive\nimmigration",
        "deflection": "false",
        "policy": "Strong borders, build the wall!",
        "response": "We need strong borders, believe me!"
    }
    
    # Patch all node functions to track execution
    with patch('political_agent_graph.graph.analyze_sentiment', 
               side_effect=lambda state: (executed_nodes.append('analyze_sentiment'), original_analyze_sentiment(state))[1]), \
         patch('political_agent_graph.graph.determine_topic', 
               side_effect=lambda state: (executed_nodes.append('determine_topic'), original_determine_topic(state))[1]), \
         patch('political_agent_graph.graph.decide_deflection', 
               side_effect=lambda state: (executed_nodes.append('decide_deflection'), original_decide_deflection(state))[1]), \
         patch('political_agent_graph.graph.generate_policy_stance', 
               side_effect=lambda state: (executed_nodes.append('generate_policy_stance'), original_generate_policy_stance(state))[1]), \
         patch('political_agent_graph.graph.format_response', 
               side_effect=lambda state: (executed_nodes.append('format_response'), original_format_response(state))[1]):
        
        # Run the conversation
        await run_conversation("What are your immigration policies?")
    
    # Check that all nodes were executed in the correct order
    assert executed_nodes == [
        'analyze_sentiment', 
        'determine_topic', 
        'decide_deflection', 
        'generate_policy_stance', 
        'format_response'
    ]


if __name__ == "__main__":
    # Run the tests
    pytest.main(["-xvs", __file__])

