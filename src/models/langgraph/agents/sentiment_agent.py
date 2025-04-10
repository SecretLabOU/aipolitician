#!/usr/bin/env python3
"""
Sentiment Agent for the AI Politician system.
This agent analyzes the sentiment of user input to determine if deflection is needed.
"""
import sys
import torch
import json
import logging
from pathlib import Path
from typing import Dict, Any
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Add the project root to the Python path
root_dir = Path(__file__).parent.parent.parent.parent.parent.absolute()
sys.path.insert(0, str(root_dir))

from src.models.langgraph.config import SENTIMENT_MODEL_ID, SENTIMENT_DEFLECTION_THRESHOLD

# Global cache for models
_sentiment_model = None
_sentiment_tokenizer = None
_sentiment_model_loading = False

# Silence the transformer logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("tokenizers").setLevel(logging.ERROR)

def _get_sentiment_model_and_tokenizer():
    """Load or get cached sentiment analysis model."""
    global _sentiment_model, _sentiment_tokenizer, _sentiment_model_loading
    
    if _sentiment_model is not None and _sentiment_tokenizer is not None:
        return _sentiment_model, _sentiment_tokenizer
    
    if _sentiment_model_loading:
        print("Sentiment model is already loading...")
        return None, None
    
    _sentiment_model_loading = True
    
    try:
        # Load sentiment model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading sentiment analysis model...")
        
        model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_ID)
        tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_ID)
        
        model = model.to(device)
        model.eval()
        
        _sentiment_model = model
        _sentiment_tokenizer = tokenizer
        _sentiment_model_loading = False
        print("Sentiment analysis model loaded successfully")
        return model, tokenizer
    
    except Exception as e:
        _sentiment_model_loading = False
        print(f"Error loading sentiment model: {str(e)}")
        print("Using simple sentiment analysis as fallback")
        return None, None

def _simple_sentiment_analysis(prompt: str) -> Dict[str, Any]:
    """Simple rule-based sentiment analysis as fallback."""
    negative_words = [
        'hate', 'awful', 'terrible', 'bad', 'worse', 'worst', 'stupid', 'idiot', 
        'incompetent', 'failure', 'fail', 'liar', 'lies', 'corrupt', 'fraud', 'cheat',
        'criminal', 'disaster', 'pathetic'
    ]
    
    prompt_lower = prompt.lower()
    negative_count = sum(1 for word in negative_words if word in prompt_lower)
    is_question = "?" in prompt
    
    # Check if this is a simple question
    question_starters = ["who", "what", "where", "when", "how", "why", "is", "are", "can", "do", "does"]
    is_simple_question = (is_question or any(prompt_lower.strip().startswith(starter) for starter in question_starters)) and len(prompt.split()) < 15
    
    # If it's a simple question with no negative words, it's neutral
    if is_simple_question and negative_count == 0:
        sentiment_score = 0.1
        sentiment_category = "neutral"
        is_biased = False
        contains_personal_attack = False
        is_gotcha_question = False
    elif negative_count >= 2:
        sentiment_score = -0.7
        sentiment_category = "negative"
        is_biased = True
        contains_personal_attack = negative_count >= 3
        is_gotcha_question = is_question and negative_count > 0
    elif negative_count == 1:
        sentiment_score = -0.3
        sentiment_category = "slightly negative" 
        is_biased = negative_count > 0
        contains_personal_attack = False
        is_gotcha_question = is_question and negative_count > 0
    else:
        sentiment_score = 0.1
        sentiment_category = "neutral"
        is_biased = False
        contains_personal_attack = False
        is_gotcha_question = False
    
    return {
        "sentiment_score": sentiment_score,
        "sentiment_category": sentiment_category,
        "is_biased": is_biased,
        "contains_personal_attack": contains_personal_attack,
        "is_gotcha_question": is_gotcha_question
    }

def analyze_sentiment_details(prompt: str, politician_name: str) -> Dict[str, Any]:
    """
    Analyze the sentiment of the user input towards the politician.
    
    Returns:
        Dict containing sentiment score, sentiment category, and whether deflection is needed
    """
    model, tokenizer = _get_sentiment_model_and_tokenizer()
    
    if model is None or tokenizer is None:
        # Fallback to simple sentiment analysis
        return _simple_sentiment_analysis(prompt)
        
    try:
        # Use the RoBERTa model for emotions
        inputs = tokenizer(prompt, truncation=True, padding=True, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = outputs.logits.softmax(dim=-1)
        
        # For RoBERTa go_emotions, we have multiple emotion categories
        # Extract the relevant ones for our analysis
        emotion_scores = predictions[0].cpu().numpy()
        
        # Get the top emotions and their scores
        emotions = tokenizer.config.id2label if hasattr(tokenizer, 'config') else {0: 'negative', 1: 'neutral', 2: 'positive'}
        emotion_data = {emotion: float(score) for emotion, score in zip(emotions.values(), emotion_scores)}
        
        # Group emotions into categories
        negative_emotions = ['anger', 'annoyance', 'disappointment', 'disapproval', 'disgust', 'grief', 'sadness', 'negative']
        positive_emotions = ['admiration', 'approval', 'caring', 'excitement', 'gratitude', 'joy', 'love', 'optimism', 'pride', 'positive']
        
        # Calculate the aggregate sentiment
        negative_score = sum(emotion_data.get(e, 0) for e in negative_emotions)
        positive_score = sum(emotion_data.get(e, 0) for e in positive_emotions)
        
        # Map to a -1 to 1 score (same range as used in the system)
        sentiment_score = float(positive_score - negative_score)
        
        # Check if this is a question (questions are often neutral)
        is_question = "?" in prompt
        
        # Simple question detection - short inputs with question marks or starting with who/what/where/when/how/why
        question_starters = ["who", "what", "where", "when", "how", "why", "is", "are", "can", "do", "does"]
        is_simple_question = (is_question or any(prompt.lower().strip().startswith(starter) for starter in question_starters)) and len(prompt.split()) < 15
        
        # Determine category based on score and question type
        if is_simple_question and abs(sentiment_score) < 0.5:
            # For simple questions, bias toward neutral unless strongly emotional
            category = "neutral"
            # Adjust sentiment score to be more neutral for simple questions
            sentiment_score = sentiment_score * 0.5  # Dampen the sentiment for questions
        elif sentiment_score < -0.3:
            category = "negative"
        elif sentiment_score < 0.1:
            category = "slightly negative"
        elif sentiment_score < 0.3:
            category = "neutral"
        else:
            category = "positive"
        
        # Determine if question contains personal attacks
        contains_personal_attack = emotion_data.get('anger', 0) > 0.3 or emotion_data.get('disgust', 0) > 0.3 or emotion_data.get('negative', 0) > 0.7
        
        # For simple questions, reduce the likelihood of detecting personal attacks
        if is_simple_question:
            contains_personal_attack = contains_personal_attack and negative_score > 0.6
        
        # Determine if question is biased
        is_biased = negative_score > 0.4
        
        # Simple questions are less likely to be biased
        if is_simple_question:
            is_biased = is_biased and negative_score > 0.6
        
        # Determine if it's a "gotcha" question
        is_gotcha = is_question and (negative_score > 0.3 or contains_personal_attack)
        
        # Simple questions are unlikely to be gotcha questions
        if is_simple_question:
            is_gotcha = is_gotcha and negative_score > 0.5
        
        return {
            "sentiment_score": sentiment_score,
            "sentiment_category": category,
            "is_biased": is_biased,
            "contains_personal_attack": contains_personal_attack,
            "is_gotcha_question": is_gotcha,
            "emotion_details": emotion_data  # Include detailed emotion analysis
        }
    
    except Exception as e:
        print(f"Error during sentiment analysis: {str(e)}")
        return _simple_sentiment_analysis(prompt)

def analyze_sentiment(state: Dict[str, Any]) -> Dict[str, Any]:
    """Process the user input to analyze sentiment and determine if deflection is needed."""
    prompt = state["user_input"]
    politician_name = state["politician_identity"].title()  # Convert "biden" to "Biden"
    
    # Analyze sentiment
    sentiment_analysis = analyze_sentiment_details(prompt, politician_name)
    
    # Remove detailed emotion data from state (keeps it cleaner)
    if "emotion_details" in sentiment_analysis:
        del sentiment_analysis["emotion_details"]
    
    # Check for basic identity or information questions
    basic_question_patterns = [
        "who are you", "what is your name", "tell me about yourself", 
        "introduce yourself", "what do you do", "what's your role", 
        "what's your job", "who is", "what is", "how are you"
    ]
    
    is_basic_question = any(pattern.lower() in prompt.lower() for pattern in basic_question_patterns)
    
    # Determine if deflection is needed based on negative sentiment and lack of knowledge
    # Multiple factors determine if deflection is needed:
    # 1. Very negative sentiment (below threshold)
    # 2. Contains personal attacks
    # 3. Is a gotcha question
    # 4. Is highly biased AND we don't have good factual information
    
    # Don't deflect on basic identity questions regardless of sentiment
    should_deflect = (
        not is_basic_question and (
            sentiment_analysis["sentiment_score"] < SENTIMENT_DEFLECTION_THRESHOLD or
            sentiment_analysis["contains_personal_attack"] or
            sentiment_analysis["is_gotcha_question"] or
            (sentiment_analysis["is_biased"] and not state.get("has_knowledge", False))
        )
    )
    
    # If this is a basic question but we're deflecting for some reason, add a note for tracing
    deflection_reason = None
    if should_deflect:
        if sentiment_analysis["sentiment_score"] < SENTIMENT_DEFLECTION_THRESHOLD:
            deflection_reason = "Negative sentiment detected"
        elif sentiment_analysis["contains_personal_attack"]:
            deflection_reason = "Contains personal attack"
        elif sentiment_analysis["is_gotcha_question"]:
            deflection_reason = "Gotcha question detected"
        elif sentiment_analysis["is_biased"] and not state.get("has_knowledge", False):
            deflection_reason = "Biased question with no supporting knowledge"
    
    # Update state with sentiment analysis
    return {
        **state,
        "sentiment_analysis": sentiment_analysis,
        "should_deflect": should_deflect,
        "deflection_reason": deflection_reason
    } 