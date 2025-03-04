"""Political agent graph implementation.

This module provides a LangGraph that simulates conversations with politicians.
It processes user input to generate authentic responses based on politicians'
speech patterns, policy positions, and rhetorical styles.
"""

import logging
import time
from typing import Dict, List, Literal, TypedDict, Union, cast, Any

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from political_agent_graph.mock_db import DB_REGISTRY, persona_manager, select_persona
from political_agent_graph.prompts import *
from political_agent_graph.state import AgentState, InputState
from political_agent_graph.config import get_model_for_task
from shared.utils import load_chat_model

logger = logging.getLogger("political_agent_graph")

class RouterResponse(TypedDict):
    """Response format for database router."""
    selected_databases: List[str]

# Core agent nodes
async def initialize_persona(state: AgentState, config: RunnableConfig) -> dict:
    """Initialize the persona for the conversation."""
    # If no persona set, select default or extract from user message
    if not state.persona_id:
        if state.messages and len(state.messages) == 1:
            message = state.messages[0].content.lower()
            # Check for persona mentions in the message
            for persona_id, persona in persona_manager.personas.items():
                if persona["name"].lower() in message:
                    select_persona(persona_id)
                    persona = persona_manager.get_active_persona()
                    state.persona_id = persona["id"]
                    state.persona_name = persona["name"]
                    state.persona_details = persona
                    break
        
        # If still no persona, use default
        if not state.persona_id:
            try:
                persona = persona_manager.get_active_persona()
                state.persona_id = persona["id"]
                state.persona_name = persona["name"]
                state.persona_details = persona
            except ValueError:
                # Default to first available persona
                if persona_manager.personas:
                    first_id = list(persona_manager.personas.keys())[0]
                    select_persona(first_id)
                    persona = persona_manager.get_active_persona()
                    state.persona_id = persona["id"]
                    state.persona_name = persona["name"]
                    state.persona_details = persona
    
    # Check for persona change request in the latest message
    elif state.messages and len(state.messages) > 1:
        last_message = state.messages[-1].content.lower()
        change_keywords = ["speak as", "talk like", "switch to", "change to", "pretend you're", "act like"]
        
        if any(keyword in last_message for keyword in change_keywords):
            for persona_id, persona in persona_manager.personas.items():
                if persona["name"].lower() in last_message:
                    select_persona(persona_id)
                    persona = persona_manager.get_active_persona()
                    state.persona_id = persona["id"]
                    state.persona_name = persona["name"]
                    state.persona_details = persona
                    state.persona_change_message = f"I'm now speaking as {persona['name']}, {persona['role']}."
                    break
    
    if not state.conversation_context:
        state.conversation_context = f"Speaking as {state.persona_name}, {state.persona_details.get('role', '')}"
    
    return {
        "persona_id": state.persona_id,
        "persona_name": state.persona_name,
        "persona_details": state.persona_details,
        "persona_change_message": state.persona_change_message if hasattr(state, "persona_change_message") else None
    }

async def analyze_sentiment(state: AgentState, config: RunnableConfig) -> dict:
    """Analyze sentiment of user input."""
    # Handle persona change notification
    if hasattr(state, "persona_change_message") and state.persona_change_message:
        return {
            "sentiment": "neutral",
            "primary_emotion": "neutral",
            "emotions": {"neutral": 1.0},
            "emotional_context": "Changing persona as requested"
        }
    
    # Analyze sentiment using LLM
    model = load_chat_model(get_model_for_task("sentiment_analysis"))
    
    persona_info = f"""Analyzing as: {state.persona_name}
Role: {state.persona_details.get('role', '')}
Party: {state.persona_details.get('party', '')}"""
    
    messages = [
        {"role": "system", "content": SENTIMENT_PROMPT},
        {"role": "human", "content": f"{persona_info}\n\nUser message: {state.messages[-1].content}"},
    ]
    
    try:
        structured_model = model.with_structured_output({
            "primary_emotion": str,
            "emotions": dict,
            "emotional_context": str,
            "summary": str
        })
        response = await structured_model.ainvoke(messages)
        
        return {
            "sentiment": response["summary"],
            "primary_emotion": response["primary_emotion"],
            "emotions": response["emotions"],
            "emotional_context": response["emotional_context"]
        }
    except Exception:
        # Fallback to simpler response
        response = await model.ainvoke(messages)
        return {"sentiment": response.content}

async def extract_context(state: AgentState, config: RunnableConfig) -> dict:
    """Extract context from user input."""
    model = load_chat_model(get_model_for_task("context_extraction"))
    
    # Get policy areas to help with context extraction
    policy_areas = ""
    if "policy_stances" in state.persona_details:
        policy_areas = ", ".join(state.persona_details["policy_stances"].keys())
    
    context_request = f"""Persona: {state.persona_name}
Key policy areas: {policy_areas}
Previous conversation context: {state.conversation_context}

User message: {state.messages[-1].content}"""
    
    messages = [
        {"role": "system", "content": CONTEXT_PROMPT},
        {"role": "human", "content": context_request},
    ]
    
    response = await model.ainvoke(messages)
    return {"context": response.content}

async def route_by_context(state: AgentState, config: RunnableConfig) -> dict:
    """Route to appropriate databases based on context."""
    model = load_chat_model(get_model_for_task("routing"))
    model = model.with_structured_output(RouterResponse)
    
    routing_request = f"""Persona: {state.persona_name}
Context from user message: {state.context}

Consider recurring topics from memory: {', '.join(state.recurring_topics) if state.recurring_topics else 'None'}"""
    
    messages = [
        {"role": "system", "content": ROUTING_PROMPT},
        {"role": "human", "content": routing_request},
    ]
    
    response = cast(RouterResponse, await model.ainvoke(messages))
    
    # Track recurring topics
    new_topic = state.context.split('\n')[0] if '\n' in state.context else state.context[:50]
    if new_topic not in state.recurring_topics:
        state.recurring_topics.append(new_topic)
        if len(state.recurring_topics) > 5:
            state.recurring_topics = state.recurring_topics[-5:]
    
    return {
        "selected_databases": response["selected_databases"], 
        "recurring_topics": state.recurring_topics
    }

async def query_databases(state: AgentState, config: RunnableConfig) -> dict:
    """Query selected databases and aggregate results."""
    results = {}
    aggregated = []
    
    for db in state.selected_databases:
        db_result = DB_REGISTRY[db](state.context)
        results[f"{db}_data"] = db_result
        aggregated.append(db_result)
    
    results["aggregated_data"] = " | ".join(aggregated)
    return results

async def generate_tone(state: AgentState, config: RunnableConfig) -> dict:
    """Generate appropriate tone for response."""
    persona_style = DB_REGISTRY["persona"]("")
    model = load_chat_model(get_model_for_task("tone_generation"))
    
    # Include emotion data if available
    emotion_info = ""
    if state.primary_emotion and state.emotions:
        emotion_details = ", ".join([f"{e}: {i:.1f}" for e, i in state.emotions.items()])
        emotion_info = f"\nEmotions: {emotion_details}\nPrimary emotion: {state.primary_emotion}"
    
    messages = [
        {"role": "system", "content": TONE_PROMPT},
        {"role": "human", "content": f"Sentiment: {state.sentiment}{emotion_info}\nData: {state.aggregated_data}\nPersona: {persona_style}"},
    ]
    
    response = await model.ainvoke(messages)
    return {"tone": response.content, "persona_style": persona_style}

async def generate_deflection(state: AgentState, config: RunnableConfig) -> dict:
    """Generate appropriate deflection when no data is found."""
    persona_style = DB_REGISTRY["persona"]("")
    model = load_chat_model(get_model_for_task("deflection"))
    
    # Include emotion data if available
    emotion_info = ""
    if state.primary_emotion and state.emotions:
        emotion_details = ", ".join([f"{e}: {i:.1f}" for e, i in state.emotions.items()])
        emotion_info = f"\nEmotions: {emotion_details}\nPrimary emotion: {state.primary_emotion}"
    
    messages = [
        {"role": "system", "content": DEFLECTION_PROMPT},
        {"role": "human", "content": f"Sentiment: {state.sentiment}{emotion_info}\nContext: {state.context}\nPersona: {persona_style}"},
    ]
    
    response = await model.ainvoke(messages)
    return {"deflection": response.content, "persona_style": persona_style}

async def compose_response(state: AgentState, config: RunnableConfig) -> dict:
    """Compose draft response combining tone/deflection with data."""
    chat_history = DB_REGISTRY["chat_memory"]("")
    model = load_chat_model(get_model_for_task("response_composition"))
    
    content = f"""Tone: {state.tone}
Data/Deflection: {state.aggregated_data or state.deflection}
Chat History: {chat_history}"""
    
    messages = [
        {"role": "system", "content": RESPONSE_PROMPT},
        {"role": "human", "content": content},
    ]
    
    response = await model.ainvoke(messages)
    return {"draft_response": response.content}

async def fact_check(state: AgentState, config: RunnableConfig) -> dict:
    """Fact check the draft response against known data."""
    facts = DB_REGISTRY["factual_kb"]("")
    model = load_chat_model(get_model_for_task("fact_checking"))
    
    messages = [
        {"role": "system", "content": FACT_CHECK_PROMPT},
        {"role": "human", "content": f"Response: {state.draft_response}\nFacts: {facts}\nData: {state.aggregated_data}"},
    ]
    
    response = await model.ainvoke(messages)
    return {"verified_response": response.content}

async def generate_final_output(state: AgentState, config: RunnableConfig) -> dict:
    """Generate final output message to user."""
    # Handle persona change message
    if hasattr(state, "persona_change_message") and state.persona_change_message:
        return {"messages": [HumanMessage(content=state.persona_change_message)]}
    
    model = load_chat_model(get_model_for_task("final_output"))
    
    messages = [
        {"role": "system", "content": FINAL_OUTPUT_PROMPT},
        {"role": "human", "content": f"Tone: {state.tone}\nPersona: {state.persona_style}\nVerified Response: {state.verified_response}"},
    ]
    
    response = await model.ainvoke(messages)
    return {"messages": [response]}

# Multi-persona features
async def generate_multi_persona_debate(state: AgentState, config: RunnableConfig) -> dict:
    """Generate a simulated debate between multiple politicians."""
    model = load_chat_model(get_model_for_task("multi_persona"))
    
    # Get current and additional personas (up to 3 others)
    personas_info = [persona_manager.get_active_persona()]
    for persona_id, persona in persona_manager.personas.items():
        if persona_id != personas_info[0]["id"] and len(personas_info) < 4:
            personas_info.append(persona)
    
    # Prepare persona details relevant to the topic
    personas_details = ""
    for persona in personas_info:
        personas_details += f"\n\n{persona['name']} ({persona['role']}, {persona['party']}):\n"
        personas_details += f"Speech style: {persona.get('speech_patterns', {}).get('sentence_structure', 'No data')}\n"
        
        # Add relevant policy positions
        context_words = state.context.lower().split()
        for policy_area, details in persona.get('policy_stances', {}).items():
            if any(word in policy_area.lower() for word in context_words):
                personas_details += f"{policy_area.capitalize()} position: {details['position']}\n"
                personas_details += f"Talking points: {', '.join(details.get('talking_points', ['No data'])[:2])}\n"
    
    messages = [
        {"role": "system", "content": MULTI_PERSONA_PROMPT},
        {"role": "human", "content": f"""Topic from user: {state.messages[-1].content}
Context of discussion: {state.context}

Participating politicians:
{personas_details}

Generate a realistic political exchange between these politicians on this topic."""},
    ]
    
    response = await model.ainvoke(messages)
    return {"messages": [response]}

async def compare_persona_positions(state: AgentState, config: RunnableConfig) -> dict:
    """Compare how different politicians would respond to the same question."""
    model = load_chat_model(get_model_for_task("multi_persona"))
    
    # Get current and additional personas (up to 3 others)
    personas_info = [persona_manager.get_active_persona()]
    for persona_id, persona in persona_manager.personas.items():
        if persona_id != personas_info[0]["id"] and len(personas_info) < 4:
            personas_info.append(persona)
    
    # Prepare relevant policy positions for each persona
    personas_details = ""
    for persona in personas_info:
        personas_details += f"\n\n{persona['name']} ({persona['role']}, {persona['party']}):\n"
        
        # Add relevant policy positions
        context_words = state.context.lower().split()
        for policy_area, details in persona.get('policy_stances', {}).items():
            if any(word in policy_area.lower() for word in context_words):
                personas_details += f"{policy_area.capitalize()} position: {details['position']}\n"
                personas_details += f"Key proposals: {', '.join(details.get('key_proposals', ['No data'])[:3])}\n"
    
    messages = [
        {"role": "system", "content": PERSONA_COMPARISON_PROMPT},
        {"role": "human", "content": f"""Question from user: {state.messages[-1].content}
Context: {state.context}

Politicians to compare:
{personas_details}

Compare how these politicians would approach this topic."""},
    ]
    
    response = await model.ainvoke(messages)
    return {"messages": [response]}

# Routing functions
def check_data_found(state: AgentState) -> Literal["generate_tone", "generate_deflection"]:
    """Determine if data was found and route accordingly."""
    return "generate_tone" if state.aggregated_data.strip() else "generate_deflection"

def route_special_modes(state: AgentState) -> Literal["generate_multi_persona_debate", "compare_persona_positions", "normal_flow"]:
    """Check if user requested special modes like debate or comparison."""
    if not state.messages:
        return "normal_flow"
        
    message = state.messages[-1].content.lower()
    
    # Check for debate request
    debate_keywords = ["debate", "conversation", "discuss", "all respond", "everyone respond", "how would they all"]
    if any(keyword in message for keyword in debate_keywords):
        return "generate_multi_persona_debate"
    
    # Check for comparison request
    comparison_keywords = ["compare", "difference", "contrast", "versus", "vs", "different perspectives", "how would each"]
    if any(keyword in message for keyword in comparison_keywords):
        return "compare_persona_positions"
    
    return "normal_flow"

def route_sentiment(state: AgentState) -> Literal["generate_tone", "generate_deflection"]:
    """Route based on sentiment analysis."""
    # Use enhanced emotion data if available
    if state.emotions:
        negative_emotions = [
            "anger", "frustration", "disgust", "disappointment", 
            "sadness", "fear", "annoyance", "hostility", "contempt"
        ]
        
        # Calculate negative emotion intensity
        negative_intensity = sum(state.emotions.get(emotion, 0) for emotion in negative_emotions)
        total_intensity = sum(state.emotions.values())
        
        # Route based on emotion intensity
        if total_intensity > 0 and (negative_intensity / total_intensity) > 0.6:
            state.response_type = "deflection_negative_ratio"
            return "generate_deflection"
        elif state.primary_emotion in negative_emotions and state.emotions.get(state.primary_emotion, 0) > 0.7:
            state.response_type = f"deflection_{state.primary_emotion}"
            return "generate_deflection"
        else:
            state.response_type = "standard_response"
            return "generate_tone"
    
    # Fallback to basic sentiment routing
    if "negative" in state.sentiment.lower():
        state.response_type = "deflection_sentiment"
        return "generate_deflection"
    else:
        state.response_type = "standard_response"
        return "generate_tone"

# Create and compile the graph
def create_political_graph() -> StateGraph:
    """Create the political agent graph."""
    builder = StateGraph(AgentState, input=InputState)
    
    # Add all nodes
    builder.add_node("initialize_persona", initialize_persona)
    builder.add_node("analyze_sentiment", analyze_sentiment)
    builder.add_node("extract_context", extract_context)
    builder.add_node("route_by_context", route_by_context)
    builder.add_node("query_databases", query_databases)
    builder.add_node("generate_tone", generate_tone)
    builder.add_node("generate_deflection", generate_deflection)
    builder.add_node("compose_response", compose_response)
    builder.add_node("fact_check", fact_check)
    builder.add_node("generate_final_output", generate_final_output)
    builder.add_node("generate_multi_persona_debate", generate_multi_persona_debate)
    builder.add_node("compare_persona_positions", compare_persona_positions)
    
    # Define graph structure
    builder.add_edge(START, "initialize_persona")
    
    # Special modes routing
    builder.add_conditional_edges(
        "initialize_persona",
        route_special_modes,
        {
            "generate_multi_persona_debate": "generate_multi_persona_debate",
            "compare_persona_positions": "compare_persona_positions",
            "normal_flow": "analyze_sentiment"
        }
    )
    
    # Special modes go directly to final output
    builder.add_edge("generate_multi_persona_debate", END)
    builder.add_edge("compare_persona_positions", END)
    
    # Normal flow
    builder.add_edge("analyze_sentiment", "extract_context")
    builder.add_edge("extract_context", "route_by_context")
    builder.add_edge("route_by_context", "query_databases")
    
    # Data-based routing
    builder.add_conditional_edges("query_databases", check_data_found)
    builder.add_conditional_edges("analyze_sentiment", route_sentiment)
    
    # Response generation flow
    builder.add_edge("generate_tone", "compose_response")
    builder.add_edge("generate_deflection", "compose_response")
    builder.add_edge("compose_response", "fact_check")
    builder.add_edge("fact_check", "generate_final_output")
    builder.add_edge("generate_final_output", END)
    
    # Compile the graph
    return builder.compile()

# Create the graph
graph = create_political_graph()
graph.name = "PoliticalAgentGraph"
