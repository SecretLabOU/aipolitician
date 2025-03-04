"""Unit tests for configuration module."""

import pytest

from political_agent_graph.config import get_model_for_task, get_temperature_for_task


def test_task_configuration():
    """Test that we get appropriate models for different tasks."""
    # Test sentiment analysis task (should use mistral)
    model = get_model_for_task("analyze_sentiment")
    assert model is not None, "Model should not be None"
    
    # Test response formatting task (should use trump)
    model = get_model_for_task("format_response")
    assert model is not None, "Model should not be None"
    
    # Test deflection decision task (should use trump)
    model = get_model_for_task("decide_deflection")
    assert model is not None, "Model should not be None"
    
    # Test unknown task (should fallback to mistral)
    model = get_model_for_task("nonexistent_task")
    assert model is not None, "Model should not be None even for unknown task"


def test_temperature_configuration():
    """Test that we get appropriate temperature settings for different tasks."""
    # Analytical tasks should have low temperature
    temp = get_temperature_for_task("analyze_sentiment")
    assert temp < 0.5, "Analytical tasks should have low temperature"
    
    # Creative tasks should have higher temperature
    temp = get_temperature_for_task("format_response")
    assert temp >= 0.5, "Creative tasks should have higher temperature"
    
    # Unknown tasks should have a default temperature
    temp = get_temperature_for_task("nonexistent_task")
    assert 0 <= temp <= 1, "Temperature should be in valid range even for unknown task"


if __name__ == "__main__":
    # Run the tests
    test_task_configuration()
    test_temperature_configuration()
    print("All unit tests passed!")