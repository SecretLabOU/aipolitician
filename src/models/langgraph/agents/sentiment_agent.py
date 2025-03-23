#!/usr/bin/env python3
"""
Sentiment Agent for the AI Politician system.
This agent analyzes the sentiment of user input to determine if deflection is needed.
"""
import sys
from pathlib import Path
from typing import Dict, Any
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# Add the project root to the Python path
root_dir = Path(__file__).parent.parent.parent.parent.parent.absolute()
sys.path.insert(0, str(root_dir))

from src.models.langgraph.config import DEFAULT_MODEL, OPENAI_API_KEY, SENTIMENT_DEFLECTION_THRESHOLD

def analyze_sentiment(prompt: str, politician_name: str) -> Dict[str, Any]:
    """
    Analyze the sentiment of the user input towards the politician.
    
    Returns:
        Dict containing sentiment score, sentiment category, and whether deflection is needed
    """
    llm = ChatOpenAI(
        model=DEFAULT_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0
    )
    
    analysis_prompt = f"""
    As a political analyst, analyze the sentiment of the following user input directed at {politician_name}.
    
    User Input: {prompt}
    
    Provide an analysis with the following information, formatted as a JSON object:
    1. Sentiment score on a scale from -1.0 (extremely negative) to 1.0 (extremely positive)
    2. Sentiment category (positive, neutral, negative, hostile)
    3. Whether the question appears biased or leading
    4. Whether the question contains personal attacks
    5. Whether the question appears to be a "gotcha" question designed to trap the politician
    
    Format your response as a valid JSON object with the keys:
    - sentiment_score (float)
    - sentiment_category (string)
    - is_biased (boolean)
    - contains_personal_attack (boolean)
    - is_gotcha_question (boolean)
    """
    
    response = llm.invoke([HumanMessage(content=analysis_prompt)])
    
    # The model will return a JSON string
    import json
    try:
        analysis = json.loads(response.content)
        # Ensure all required fields are present
        required_fields = ["sentiment_score", "sentiment_category", "is_biased", 
                         "contains_personal_attack", "is_gotcha_question"]
        for field in required_fields:
            if field not in analysis:
                analysis[field] = False if field.startswith("is_") or field.startswith("contains_") else 0.0
        
        return analysis
    except json.JSONDecodeError:
        # Fallback if JSON parsing fails
        return {
            "sentiment_score": 0.0,
            "sentiment_category": "neutral",
            "is_biased": False,
            "contains_personal_attack": False,
            "is_gotcha_question": False
        }

def process_sentiment(state: Dict[str, Any]) -> Dict[str, Any]:
    """Process the user input to analyze sentiment and determine if deflection is needed."""
    prompt = state["user_input"]
    politician_name = state["politician_identity"].title()  # Convert "biden" to "Biden"
    
    # Analyze sentiment
    sentiment_analysis = analyze_sentiment(prompt, politician_name)
    
    # Determine if deflection is needed based on negative sentiment and lack of knowledge
    # Multiple factors determine if deflection is needed:
    # 1. Very negative sentiment (below threshold)
    # 2. Contains personal attacks
    # 3. Is a gotcha question
    # 4. Is highly biased AND we don't have good factual information
    
    should_deflect = (
        sentiment_analysis["sentiment_score"] < SENTIMENT_DEFLECTION_THRESHOLD or
        sentiment_analysis["contains_personal_attack"] or
        sentiment_analysis["is_gotcha_question"] or
        (sentiment_analysis["is_biased"] and not state.get("has_knowledge", False))
    )
    
    # Update state with sentiment analysis
    return {
        **state,
        "sentiment_analysis": sentiment_analysis,
        "should_deflect": should_deflect
    } 