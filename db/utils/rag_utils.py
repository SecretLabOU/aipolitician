"""
Utility functions for integrating databases with the RAG system.

This module provides functionality for retrieving information from
the databases based on user queries and formatting it for use in
the RAG system.
"""
from typing import Dict, List, Optional, Any, Union

from db.database import get_database


def retrieve_context_for_query(
    query: str, 
    politician: str,
    top_k: int = 3,
    search_all: bool = True
) -> str:
    """
    Retrieve context from the databases for a query.
    
    Args:
        query: The user's query
        politician: The politician's name (e.g., "Donald Trump", "Joe Biden")
        top_k: Number of results to retrieve from each database
        search_all: Whether to search all databases or only relevant ones
        
    Returns:
        A formatted string containing the retrieved context
    """
    # Map of politician names to lowercase normalized versions
    politician_map = {
        "donald trump": "Donald Trump",
        "trump": "Donald Trump",
        "joe biden": "Joe Biden",
        "biden": "Joe Biden",
    }
    
    # Normalize politician name
    norm_politician = politician.lower()
    if norm_politician in politician_map:
        politician = politician_map[norm_politician]
    
    # Determine which databases to search based on the query
    databases_to_search = []
    
    # If search_all is True, search all databases
    if search_all:
        databases_to_search = [
            'biography',
            'policy',
            'voting_record',
            'public_statements',
            'fact_check',
            'timeline',
            'legislative',
            'campaign_promises',
            'executive_actions',
            'media_coverage',
            'public_opinion',
            'controversies',
            'policy_comparison',
            'judicial_appointments',
            'foreign_policy',
            'economic_metrics',
            'charity',
        ]
    else:
        # Determine which databases to search based on the query
        query_lower = query.lower()
        
        # Add databases based on keywords in the query
        if any(kw in query_lower for kw in ["bio", "born", "family", "education", "career"]):
            databases_to_search.append('biography')
        
        if any(kw in query_lower for kw in ["policy", "position", "stance", "opinion"]):
            databases_to_search.append('policy')
        
        if any(kw in query_lower for kw in ["vote", "voting", "voted"]):
            databases_to_search.append('voting_record')
        
        if any(kw in query_lower for kw in ["speech", "statement", "interview", "said"]):
            databases_to_search.append('public_statements')
        
        if any(kw in query_lower for kw in ["fact", "check", "true", "false", "claim"]):
            databases_to_search.append('fact_check')
        
        if any(kw in query_lower for kw in ["timeline", "when", "date", "year"]):
            databases_to_search.append('timeline')
        
        if any(kw in query_lower for kw in ["bill", "legislation", "law", "sponsor"]):
            databases_to_search.append('legislative')
        
        if any(kw in query_lower for kw in ["promise", "campaign", "pledge"]):
            databases_to_search.append('campaign_promises')
        
        if any(kw in query_lower for kw in ["executive", "order", "action", "memorandum"]):
            databases_to_search.append('executive_actions')
        
        if any(kw in query_lower for kw in ["news", "media", "press", "coverage"]):
            databases_to_search.append('media_coverage')
        
        if any(kw in query_lower for kw in ["poll", "approval", "rating", "popularity"]):
            databases_to_search.append('public_opinion')
        
        if any(kw in query_lower for kw in ["controversy", "scandal", "investigation"]):
            databases_to_search.append('controversies')
        
        if any(kw in query_lower for kw in ["compare", "difference", "versus", "vs"]):
            databases_to_search.append('policy_comparison')
        
        if any(kw in query_lower for kw in ["judge", "justice", "court", "judicial", "nominate"]):
            databases_to_search.append('judicial_appointments')
        
        if any(kw in query_lower for kw in ["foreign", "international", "country", "diplomacy"]):
            databases_to_search.append('foreign_policy')
        
        if any(kw in query_lower for kw in ["economy", "economic", "gdp", "unemployment", "inflation"]):
            databases_to_search.append('economic_metrics')
        
        if any(kw in query_lower for kw in ["charity", "donation", "philanthropy", "foundation"]):
            databases_to_search.append('charity')
        
        # If no specific databases matched, search biography and policy as defaults
        if not databases_to_search:
            databases_to_search = ['biography', 'policy']
    
    # Collect all results
    all_results = []
    
    # Search each database
    for db_name in databases_to_search:
        try:
            db = get_database(db_name)
            
            # Use the search method if the database has one
            if hasattr(db, 'search_biography'):
                results = db.search_biography(query, top_k=top_k)
                all_results.extend(
                    {
                        "database": db_name,
                        "text": result["text"],
                        "score": result["score"],
                        "metadata": result["metadata"]
                    }
                    for result in results
                )
        except (ImportError, ValueError):
            # Skip databases that are not implemented yet
            pass
    
    # Sort results by score
    all_results.sort(key=lambda x: x["score"], reverse=True)
    
    # Format the results
    if not all_results:
        return "No information found."
    
    # Format the top results
    formatted_context = "### RETRIEVED CONTEXT ###\n\n"
    
    for i, result in enumerate(all_results[:5]):  # Limit to top 5 results
        source = result.get("metadata", {}).get("source_url", "")
        source_text = f"(Source: {source})" if source else ""
        
        formatted_context += f"[{i+1}] {result['text'][:500]}... {source_text}\n\n"
    
    return formatted_context


def integrate_with_chat(
    query: str,
    politician: str,
    response_template: str = "Based on the available information, {politician} {response}"
) -> str:
    """
    Integrate database retrieval with the chat system.
    
    Args:
        query: The user's query
        politician: The politician to focus on
        response_template: Template for formatting the response
        
    Returns:
        The formatted context for the RAG system
    """
    # Retrieve context from the databases
    context = retrieve_context_for_query(query, politician)
    
    # Format as a system message for the chat model
    system_message = f"""
You are answering as {politician}. Use ONLY the following retrieved information to answer the user's question. 
If the information is not sufficient to answer, acknowledge what you know and what you don't know.
Be concise and stay in character as {politician} at all times.

{context}

Remember to respond as {politician} would, using their speaking style, mannerisms, and political perspectives.
"""
    
    return system_message