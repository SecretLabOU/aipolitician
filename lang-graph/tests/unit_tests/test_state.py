"""Unit tests for the state module.

Tests for the ConversationState class and related functions.
"""

import pytest
from typing import Dict, Any

from political_agent_graph.state import ConversationState, get_initial_state


@pytest.fixture
def empty_state() -> ConversationState:
    """Fixture for an empty ConversationState."""
    return ConversationState()


@pytest.fixture
def populated_state() -> ConversationState:
    """Fixture for a populated ConversationState with sample data."""
    state = ConversationState(
        user_input="What about climate change?",
        current_topic="climate change",
        topic_sentiment="neutral",
        should_deflect=False,
        policy_stance="I believe we need to address climate change",
        final_response="Climate change is an important issue that requires action."
    )
    state.add_to_history("User", "What about climate change?")
    state.add_to_history("AI", "Climate change is an important issue that requires action.")
    return state


@pytest.fixture
def sample_state_dict() -> Dict[str, Any]:
    """Fixture for a sample state dictionary."""
    return {
        "user_input": "What's your stance on immigration?",
        "conversation_history": [
            {"speaker": "User", "text": "What's your stance on immigration?"},
            {"speaker": "AI", "text": "I support comprehensive immigration reform."}
        ],
        "current_topic": "immigration",
        "topic_sentiment": "questioning",
        "should_deflect": False,
        "deflection_topic": None,
        "policy_stance": "comprehensive immigration reform",
        "final_response": "I support comprehensive immigration reform."
    }


def test_to_dict_empty_state(empty_state: ConversationState) -> None:
    """Test to_dict() with an empty state."""
    result = empty_state.to_dict()
    
    assert isinstance(result, dict)
    assert result["user_input"] == ""
    assert result["conversation_history"] == []
    assert result["current_topic"] is None
    assert result["topic_sentiment"] is None
    assert result["should_deflect"] is False
    assert result["deflection_topic"] is None
    assert result["policy_stance"] is None
    assert result["final_response"] is None


def test_to_dict_populated_state(populated_state: ConversationState) -> None:
    """Test to_dict() with a populated state."""
    result = populated_state.to_dict()
    
    assert isinstance(result, dict)
    assert result["user_input"] == "What about climate change?"
    assert len(result["conversation_history"]) == 2
    assert result["conversation_history"][0]["speaker"] == "User"
    assert result["conversation_history"][0]["text"] == "What about climate change?"
    assert result["current_topic"] == "climate change"
    assert result["topic_sentiment"] == "neutral"
    assert result["should_deflect"] is False
    assert result["policy_stance"] == "I believe we need to address climate change"
    assert result["final_response"] == "Climate change is an important issue that requires action."


def test_from_dict(sample_state_dict: Dict[str, Any]) -> None:
    """Test from_dict() creates the correct ConversationState."""
    state = ConversationState.from_dict(sample_state_dict)
    
    assert state.user_input == "What's your stance on immigration?"
    assert len(state.conversation_history) == 2
    assert state.conversation_history[0]["speaker"] == "User"
    assert state.conversation_history[0]["text"] == "What's your stance on immigration?"
    assert state.conversation_history[1]["speaker"] == "AI"
    assert state.conversation_history[1]["text"] == "I support comprehensive immigration reform."
    assert state.current_topic == "immigration"
    assert state.topic_sentiment == "questioning"
    assert state.should_deflect is False
    assert state.deflection_topic is None
    assert state.policy_stance == "comprehensive immigration reform"
    assert state.final_response == "I support comprehensive immigration reform."


def test_from_dict_with_missing_fields() -> None:
    """Test from_dict() with missing fields uses defaults."""
    partial_data = {
        "user_input": "Hello",
        # Missing other fields
    }
    
    state = ConversationState.from_dict(partial_data)
    
    assert state.user_input == "Hello"
    assert state.conversation_history == []
    assert state.current_topic is None
    assert state.should_deflect is False


def test_add_to_history(empty_state: ConversationState) -> None:
    """Test add_to_history() appends messages correctly."""
    # Initial state should have empty history
    assert len(empty_state.conversation_history) == 0
    
    # Add user message
    empty_state.add_to_history("User", "Hello there")
    assert len(empty_state.conversation_history) == 1
    assert empty_state.conversation_history[0]["speaker"] == "User"
    assert empty_state.conversation_history[0]["text"] == "Hello there"
    
    # Add AI response
    empty_state.add_to_history("AI", "Hi, how can I help?")
    assert len(empty_state.conversation_history) == 2
    assert empty_state.conversation_history[1]["speaker"] == "AI"
    assert empty_state.conversation_history[1]["text"] == "Hi, how can I help?"


def test_get_initial_state() -> None:
    """Test get_initial_state creates proper initial state."""
    user_input = "What's your position on taxes?"
    state = get_initial_state(user_input)
    
    assert state.user_input == user_input
    assert len(state.conversation_history) == 1
    assert state.conversation_history[0]["speaker"] == "User"
    assert state.conversation_history[0]["text"] == user_input
    assert state.current_topic is None
    assert state.topic_sentiment is None
    assert state.should_deflect is False
    assert state.deflection_topic is None
    assert state.policy_stance is None
    assert state.final_response is None


def test_round_trip_serialization(populated_state: ConversationState) -> None:
    """Test round-trip serialization (to_dict -> from_dict)."""
    # Convert to dict
    state_dict = populated_state.to_dict()
    
    # Create new state from dict
    new_state = ConversationState.from_dict(state_dict)
    
    # Verify the states match
    assert new_state.user_input == populated_state.user_input
    assert new_state.conversation_history == populated_state.conversation_history
    assert new_state.current_topic == populated_state.current_topic
    assert new_state.topic_sentiment == populated_state.topic_sentiment
    assert new_state.should_deflect == populated_state.should_deflect
    assert new_state.deflection_topic == populated_state.deflection_topic
    assert new_state.policy_stance == populated_state.policy_stance
    assert new_state.final_response == populated_state.final_response

