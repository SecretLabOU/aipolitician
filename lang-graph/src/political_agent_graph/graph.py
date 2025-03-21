"""Political agent graph implementation.

This module implements the LangGraph for simulating politicians.
"""

import json
import asyncio
from typing import Dict, List, Tuple, Any, Annotated, TypedDict, Union
from dataclasses import asdict

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain.schema import HumanMessage, AIMessage

from political_agent_graph.state import ConversationState, get_initial_state
from political_agent_graph.config import get_model_for_task, get_temperature_for_task
from political_agent_graph.prompts import (
    analyze_sentiment_template,
    determine_topic_template,
    decide_deflection_template,
    generate_policy_stance_template,
    format_response_template,
)
from political_agent_graph import persona_manager


def analyze_sentiment(state: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze the sentiment of the user's input."""
    model = get_model_for_task("analyze_sentiment")
    
    prompt = analyze_sentiment_template.format(
        user_input=state["user_input"]
    )
    
    # Get sentiment from model
    sentiment = model.invoke(prompt).strip().lower()
    
    # Validate and normalize sentiment
    if sentiment not in ["positive", "negative", "neutral"]:
        # Default to neutral if model returns invalid sentiment
        sentiment = "neutral"
    
    # Update state
    state["topic_sentiment"] = sentiment
    return state


def determine_topic(state: Dict[str, Any]) -> Dict[str, Any]:
    """Determine the topic of the user's input."""
    model = get_model_for_task("determine_topic")
    
    prompt = determine_topic_template.format(
        user_input=state["user_input"]
    )
    
    # Get topic from model
    topic = model.invoke(prompt).strip().lower()
    
    # Update state
    state["current_topic"] = topic
    return state


def decide_deflection(state: Dict[str, Any]) -> Dict[str, Any]:
    """Decide whether to deflect the question."""
    model = get_model_for_task("decide_deflection")
    
    # Get active persona
    persona = persona_manager.get_active_persona()
    
    prompt = decide_deflection_template.format(
        politician_name=persona["name"],
        politician_party=persona["party"],
        user_input=state["user_input"],
        current_topic=state["current_topic"],
        topic_sentiment=state["topic_sentiment"],
        rhetoric_style=json.dumps(persona["rhetorical_style"], indent=2)
    )
    
    # Get deflection decision from model
    result = model.invoke(prompt).strip()
    
    # Parse result
    lines = result.split("\n")
    should_deflect = lines[0].lower() == "true"
    
    # Update state
    state["should_deflect"] = should_deflect
    
    # If there's a deflection topic suggestion, use it
    if should_deflect and len(lines) > 1:
        state["deflection_topic"] = lines[1].strip()
    
    return state


def generate_policy_stance(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate the politician's policy stance."""
    model = get_model_for_task("generate_policy_stance")
    
    # Get active persona
    persona = persona_manager.get_active_persona()
    
    # Determine which topic to use
    topic = state["deflection_topic"] if state["should_deflect"] else state["current_topic"]
    
    # Get the relevant policy stance from the persona data
    policy_data = persona.get("policy_stances", {})
    relevant_policy = policy_data.get(topic, {})
    
    prompt = generate_policy_stance_template.format(
        politician_name=persona["name"],
        politician_party=persona["party"],
        current_topic=topic,
        user_input=state["user_input"],
        policy_stances=json.dumps(relevant_policy, indent=2) if relevant_policy else "No specific stance on this topic.",
        speech_patterns=json.dumps(persona["speech_patterns"], indent=2)
    )
    
    # Get policy stance from model
    policy_stance = model.invoke(prompt).strip()
    
    # Update state
    state["policy_stance"] = policy_stance
    return state


def format_response(state: Dict[str, Any]) -> Dict[str, Any]:
    """Format the politician's response."""
    model = get_model_for_task("format_response")
    
    # Get active persona
    persona = persona_manager.get_active_persona()
    
    # Format conversation history for prompt
    history_text = ""
    for message in state["conversation_history"][-5:]:  # Only use last 5 messages
        history_text += f"{message['speaker']}: {message['text']}\n"
    
    prompt = format_response_template.format(
        politician_name=persona["name"],
        politician_party=persona["party"],
        user_input=state["user_input"],
        current_topic=state["current_topic"],
        should_deflect=state["should_deflect"],
        deflection_topic=state.get("deflection_topic", ""),
        policy_stance=state.get("policy_stance", ""),
        speech_patterns=json.dumps(persona["speech_patterns"], indent=2),
        rhetoric_style=json.dumps(persona["rhetorical_style"], indent=2),
        conversation_history=history_text
    )
    
    # Get formatted response from model
    response = model.invoke(prompt).strip()
    
    # Update state
    state["final_response"] = response
    state["conversation_history"].append({
        "speaker": persona["name"],
        "text": response
    })
    return state


# Build the graph
def build_graph() -> StateGraph:
    """Build the LangGraph for the political agent."""
    # Define a new graph
    builder = StateGraph(Dict)
    
    # Add nodes
    builder.add_node("analyze_sentiment", analyze_sentiment)
    builder.add_node("determine_topic", determine_topic)
    builder.add_node("decide_deflection", decide_deflection)
    builder.add_node("generate_policy_stance", generate_policy_stance)
    builder.add_node("format_response", format_response)
    
    # Add edges
    builder.add_edge("analyze_sentiment", "determine_topic")
    builder.add_edge("determine_topic", "decide_deflection")
    builder.add_edge("decide_deflection", "generate_policy_stance")
    builder.add_edge("generate_policy_stance", "format_response")
    builder.add_edge("format_response", END)
    
    # Set the entry point
    builder.set_entry_point("analyze_sentiment")
    
    # Compile the graph
    return builder.compile()


# Create the graph
graph = build_graph()


async def run_conversation(user_input: str) -> str:
    """Run a conversation turn through the graph.
    
    Args:
        user_input: The user's input message
        
    Returns:
        The politician's response
    """
    # Create initial state
    initial_state = get_initial_state(user_input)
    
    # Convert state to dict for graph processing
    state_dict = asdict(initial_state)
    
    # Run the graph
    result = await graph.ainvoke(state_dict)
    
    # Return the final response
    return result["final_response"]