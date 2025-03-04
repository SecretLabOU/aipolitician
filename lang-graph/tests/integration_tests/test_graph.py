"""Integration tests for the political agent graph."""

import pytest
import asyncio

from political_agent_graph import select_persona, run_conversation


async def test_trump_response():
    """Test that Trump responds appropriately to a policy question."""
    # Set the persona to Trump
    select_persona("donald_trump")
    
    # Test a policy question
    response = await run_conversation("What is your view on immigration?")
    
    # Check for expected phrases or content
    assert response, "Response should not be empty"
    assert len(response) > 50, "Response should be substantial"
    
    # Trump's common phrases or policy positions should be present
    immigration_terms = [
        "border", "wall", "illegal", "Mexico", "security", "legal", "country", 
        "America", "great", "tremendous", "disaster"
    ]
    
    # Count how many of Trump's typical terms appear
    term_count = sum(1 for term in immigration_terms if term.lower() in response.lower())
    
    # Should include at least a few of his typical terms
    assert term_count >= 3, f"Response lacks Trump's typical language (found {term_count} terms)"


async def test_persona_switching():
    """Test that we can switch between personas."""
    # Start with Trump
    select_persona("donald_trump")
    trump_response = await run_conversation("What do you think about climate change?")
    
    # Check if we get a valid response
    assert trump_response, "Trump response should not be empty"
    
    # In a full implementation, we would switch to other personas like:
    # select_persona("bernie_sanders")
    # sanders_response = await run_conversation("What do you think about climate change?")
    # assert sanders_response != trump_response, "Different personas should give different responses"


if __name__ == "__main__":
    # Run the tests
    asyncio.run(test_trump_response())
    asyncio.run(test_persona_switching())
    print("All tests passed!")