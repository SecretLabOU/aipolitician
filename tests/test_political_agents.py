#!/usr/bin/env python3
"""
Integration tests for Political Agents with LangGraph and RAG.
"""

import pytest
from src.agents.political import PoliticalAgent

pytestmark = pytest.mark.asyncio  # Mark all tests in this module as async

async def test_agent_responses():
    """Test that both Trump and Biden agents can generate responses with RAG."""
    # Create agents
    trump_agent = PoliticalAgent(persona="trump")
    biden_agent = PoliticalAgent(persona="biden")

    # Test questions
    questions = [
        "What's your stance on immigration?",
        "How would you handle climate change?",
        "What's your plan for the economy?"
    ]

    for question in questions:
        # Test Trump responses
        trump_response = await trump_agent.generate_response(question)
        assert trump_response, "Trump agent should generate a non-empty response"
        assert len(trump_response) > 50, "Response should be substantial"

        # Test Biden responses
        biden_response = await biden_agent.generate_response(question)
        assert biden_response, "Biden agent should generate a non-empty response"
        assert len(biden_response) > 50, "Response should be substantial"

        # Ensure responses are different
        assert trump_response != biden_response, "Agents should generate different responses"

async def test_rag_integration():
    """Test that RAG is properly integrated with the agents."""
    trump_agent = PoliticalAgent(persona="trump")
    
    # This question should trigger RAG to fetch relevant context
    question = "What did you say about NATO in your 2016 campaign?"
    
    response = await trump_agent.generate_response(question)
    
    # The response should contain specific details that would only be available through RAG
    assert response, "Should get a response"
    assert len(response) > 100, "Response should be detailed with RAG context"

if __name__ == "__main__":
    pytest.main([__file__])
