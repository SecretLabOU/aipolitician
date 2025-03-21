"""RAG utilities for the political agent system."""
from typing import Optional
import json
from pathlib import Path

def integrate_with_chat(query: str, persona_name: str) -> str:
    """Retrieve relevant context for a given query and persona.
    
    Args:
        query: The user's question
        persona_name: Name of the political persona ("Donald Trump" or "Joe Biden")
        
    Returns:
        str: Retrieved context or empty string if no relevant context found
    """
    try:
        # For now, return a simple context placeholder
        # In a real implementation, this would query a vector database
        context_data = {
            "Donald Trump": {
                "immigration": "We need strong borders, very strong borders. We're building the wall.",
                "economy": "We have the best economy in history. The stock market is at record highs.",
                "nato": "NATO needs to pay their fair share. Many countries aren't paying what they should.",
            },
            "Joe Biden": {
                "immigration": "We need comprehensive immigration reform that upholds our values.",
                "economy": "When the middle class does well, everybody does well.",
                "nato": "NATO is the strongest alliance in history. We must strengthen our partnerships.",
            }
        }
        
        # Simple keyword matching
        keywords = query.lower().split()
        for topic, response in context_data.get(persona_name, {}).items():
            if topic in keywords:
                return f"Previous statements on {topic}: {response}"
        
        return ""  # No relevant context found
        
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return "" 