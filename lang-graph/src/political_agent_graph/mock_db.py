"""Mock database implementations for the political agent graph.

This module provides mock implementations of the various databases used in the graph.
Each function simulates a database query and returns predefined data.
"""

def query_voting_db(query: str) -> str:
    """Mock implementation of voting database."""
    return "Mock voting record data: Voted yes on Bill 123, no on Bill 456"

def query_bio_db(query: str) -> str:
    """Mock implementation of biography database."""
    return "Mock biography data: Born in Springfield, graduated from State University"

def query_social_db(query: str) -> str:
    """Mock implementation of social media database."""
    return "Mock social media data: Recent tweets about climate change and education"

def query_policy_db(query: str) -> str:
    """Mock implementation of policy database."""
    return "Mock policy data: Supports renewable energy and education reform"

def query_persona_db(query: str) -> str:
    """Mock implementation of persona database."""
    return "Mock persona data: Professional and diplomatic communication style"

def query_chat_memory_db(query: str) -> str:
    """Mock implementation of chat memory database."""
    return "Mock chat history: Previously discussed education policy"

def query_factual_kb(query: str) -> str:
    """Mock implementation of factual knowledge base."""
    return "Mock fact check: Statement verified against public records"

# Dictionary mapping database names to their query functions
DB_REGISTRY = {
    "voting": query_voting_db,
    "bio": query_bio_db,
    "social": query_social_db,
    "policy": query_policy_db,
    "persona": query_persona_db,
    "chat_memory": query_chat_memory_db,
    "factual_kb": query_factual_kb,
}
