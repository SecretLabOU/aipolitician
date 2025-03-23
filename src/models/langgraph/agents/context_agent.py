#!/usr/bin/env python3
"""
Context Agent for the AI Politician system.
This agent extracts important information from user input and uses RAG to look through the knowledge base.
"""
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# Add the project root to the Python path
root_dir = Path(__file__).parent.parent.parent.parent.parent.absolute()
sys.path.insert(0, str(root_dir))

from src.models.langgraph.config import DEFAULT_MODEL, OPENAI_API_KEY, HAS_RAG

# Import RAG utilities if available
if HAS_RAG:
    from src.data.db.utils.rag_utils import integrate_with_chat

def extract_context_from_prompt(prompt: str, politician_name: str) -> str:
    """Extract key topics and context from the user prompt."""
    llm = ChatOpenAI(
        model=DEFAULT_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0
    )
    
    extraction_prompt = f"""
    As a political analyst, analyze the following user input directed at {politician_name}
    and extract the key topics, policy areas, and factual questions being asked.
    Focus on identifying specific topics that would require factual knowledge or policy positions.
    
    User Input: {prompt}
    
    Provide a concise analysis that identifies:
    1. Main topic(s)
    2. Specific policy areas mentioned
    3. Any factual claims that need verification
    4. Key entities mentioned (people, places, events)
    
    Your analysis should be structured and concise, as it will be used for retrieving relevant information.
    """
    
    response = llm.invoke([HumanMessage(content=extraction_prompt)])
    return response.content

def get_rag_context(prompt: str, politician_name: str) -> Optional[str]:
    """Get context from the RAG system if available."""
    if HAS_RAG:
        return integrate_with_chat(prompt, politician_name)
    else:
        # Simulate RAG response for testing
        return f"Simulated knowledge base information about: {extract_context_from_prompt(prompt, politician_name)}"

def process_context(state: Dict[str, Any]) -> Dict[str, Any]:
    """Process the user input to extract context and retrieve relevant information."""
    prompt = state["user_input"]
    politician_name = state["politician_identity"].title()  # Convert "biden" to "Biden"
    
    # Extract context from prompt for better retrieval
    extracted_context = extract_context_from_prompt(prompt, politician_name)
    
    # Get context from knowledge base using RAG
    rag_context = get_rag_context(prompt, politician_name) if state.get("use_rag", True) else None
    
    # Combine both contexts
    combined_context = f"Extracted Topics: {extracted_context}\n\n"
    if rag_context:
        combined_context += f"Knowledge Base Context: {rag_context}"
    
    # Update state with context
    return {
        **state,
        "context": combined_context,
        "has_knowledge": bool(rag_context and rag_context != f"Simulated knowledge base information about: {extracted_context}")
    } 