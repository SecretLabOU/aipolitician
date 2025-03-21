#!/usr/bin/env python3
"""
Integration tests for Political Agents with LangGraph and RAG.
"""

import pytest
from src.agents.political import PoliticalAgent

pytestmark = pytest.mark.asyncio  # Mark all tests in this module as async

@pytest.mark.asyncio
async def test_agent_responses():
    """Test that both Trump and Biden agents can generate responses."""
    # Create agents
    trump_agent = PoliticalAgent(persona="trump")
    biden_agent = PoliticalAgent(persona="biden")

    # Test questions
    questions = [
        "What's your stance on immigration?",
        "How would you handle NATO?",
        "What's your plan for the economy?"
    ]

    for question in questions:
        # Test Trump responses
        trump_response = await trump_agent.generate_response(question)
        assert trump_response, "Trump agent should generate a non-empty response"
        
        # Test Biden responses
        biden_response = await biden_agent.generate_response(question)
        assert biden_response, "Biden agent should generate a non-empty response"
        
        # Ensure responses are different
        assert trump_response != biden_response, "Agents should generate different responses"

@pytest.mark.asyncio
async def test_rag_integration():
    """Test that RAG integration works as expected."""
    trump_agent = PoliticalAgent(persona="trump")
    
    # Test with a question that should trigger RAG
    question = "What is your position on NATO?"
    response = await trump_agent.generate_response(question)
    
    assert "NATO" in response.upper(), "Response should include RAG context about NATO"
    assert response, "Should get a response with RAG integration"

if __name__ == "__main__":
    pytest.main([__file__])
