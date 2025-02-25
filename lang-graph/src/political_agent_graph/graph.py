"""Main implementation of the political agent graph.

This module implements the flowchart as a LangGraph, defining the nodes and edges
that process user input through sentiment analysis, context extraction, database queries,
and response generation.
"""

from typing import List, Literal, TypedDict, cast

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from political_agent_graph.mock_db import DB_REGISTRY
from political_agent_graph.prompts import (
    CONTEXT_PROMPT,
    DEFLECTION_PROMPT,
    FACT_CHECK_PROMPT,
    FINAL_OUTPUT_PROMPT,
    RESPONSE_PROMPT,
    ROUTING_PROMPT,
    SENTIMENT_PROMPT,
    TONE_PROMPT,
)
from political_agent_graph.state import AgentState, InputState
from shared.utils import load_chat_model


async def analyze_sentiment(state: AgentState, config: RunnableConfig) -> dict[str, str]:
    """Analyze sentiment of user input."""
    model = load_chat_model("anthropic/claude-3-haiku-20240307")
    messages = [
        {"role": "system", "content": SENTIMENT_PROMPT},
        {"role": "human", "content": state.messages[-1].content},
    ]
    response = await model.ainvoke(messages)
    return {"sentiment": response.content}


async def extract_context(state: AgentState, config: RunnableConfig) -> dict[str, str]:
    """Extract context from user input."""
    model = load_chat_model("anthropic/claude-3-haiku-20240307")
    messages = [
        {"role": "system", "content": CONTEXT_PROMPT},
        {"role": "human", "content": state.messages[-1].content},
    ]
    response = await model.ainvoke(messages)
    return {"context": response.content}


class RouterResponse(TypedDict):
    """Response format for the router."""
    selected_databases: List[str]


async def route_by_context(state: AgentState, config: RunnableConfig) -> dict[str, list[str]]:
    """Route to appropriate databases based on context."""
    model = load_chat_model("anthropic/claude-3-haiku-20240307")
    model = model.with_structured_output(RouterResponse)
    messages = [
        {"role": "system", "content": ROUTING_PROMPT},
        {"role": "human", "content": state.context},
    ]
    response = cast(RouterResponse, await model.ainvoke(messages))
    return {"selected_databases": response["selected_databases"]}


async def query_databases(state: AgentState, config: RunnableConfig) -> dict[str, str]:
    """Query selected databases and aggregate results."""
    results = {}
    aggregated = []
    
    for db in state.selected_databases:
        if db == "voting":
            results["voting_data"] = DB_REGISTRY["voting"](state.context)
            aggregated.append(results["voting_data"])
        elif db == "bio":
            results["bio_data"] = DB_REGISTRY["bio"](state.context)
            aggregated.append(results["bio_data"])
        elif db == "social":
            results["social_data"] = DB_REGISTRY["social"](state.context)
            aggregated.append(results["social_data"])
        elif db == "policy":
            results["policy_data"] = DB_REGISTRY["policy"](state.context)
            aggregated.append(results["policy_data"])
    
    results["aggregated_data"] = " | ".join(aggregated)
    return results


def check_data_found(state: AgentState) -> Literal["generate_tone", "generate_deflection"]:
    """Check if data was found and route accordingly."""
    return "generate_tone" if state.aggregated_data.strip() else "generate_deflection"


async def generate_tone(state: AgentState, config: RunnableConfig) -> dict[str, str]:
    """Generate appropriate tone for response."""
    # Get persona style
    persona_style = DB_REGISTRY["persona"]("")
    
    model = load_chat_model("anthropic/claude-3-haiku-20240307")
    messages = [
        {"role": "system", "content": TONE_PROMPT},
        {"role": "human", "content": f"Sentiment: {state.sentiment}\nData: {state.aggregated_data}\nPersona: {persona_style}"},
    ]
    response = await model.ainvoke(messages)
    return {"tone": response.content, "persona_style": persona_style}


async def generate_deflection(state: AgentState, config: RunnableConfig) -> dict[str, str]:
    """Generate appropriate deflection."""
    # Get persona style
    persona_style = DB_REGISTRY["persona"]("")
    
    model = load_chat_model("anthropic/claude-3-haiku-20240307")
    messages = [
        {"role": "system", "content": DEFLECTION_PROMPT},
        {"role": "human", "content": f"Sentiment: {state.sentiment}\nContext: {state.context}\nPersona: {persona_style}"},
    ]
    response = await model.ainvoke(messages)
    return {"deflection": response.content, "persona_style": persona_style}


async def compose_response(state: AgentState, config: RunnableConfig) -> dict[str, str]:
    """Compose draft response."""
    # Get chat history
    chat_history = DB_REGISTRY["chat_memory"]("")
    
    model = load_chat_model("anthropic/claude-3-haiku-20240307")
    content = f"""Tone: {state.tone}
Data/Deflection: {state.aggregated_data or state.deflection}
Chat History: {chat_history}"""
    
    messages = [
        {"role": "system", "content": RESPONSE_PROMPT},
        {"role": "human", "content": content},
    ]
    response = await model.ainvoke(messages)
    return {"draft_response": response.content}


async def fact_check(state: AgentState, config: RunnableConfig) -> dict[str, str]:
    """Fact check the draft response."""
    # Get factual knowledge
    facts = DB_REGISTRY["factual_kb"]("")
    
    model = load_chat_model("anthropic/claude-3-haiku-20240307")
    messages = [
        {"role": "system", "content": FACT_CHECK_PROMPT},
        {"role": "human", "content": f"Response: {state.draft_response}\nFacts: {facts}\nData: {state.aggregated_data}"},
    ]
    response = await model.ainvoke(messages)
    return {"verified_response": response.content}


async def generate_final_output(
    state: AgentState, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate final output."""
    model = load_chat_model("anthropic/claude-3-haiku-20240307")
    messages = [
        {"role": "system", "content": FINAL_OUTPUT_PROMPT},
        {
            "role": "human",
            "content": f"""Tone: {state.tone}
Persona: {state.persona_style}
Verified Response: {state.verified_response}""",
        },
    ]
    response = await model.ainvoke(messages)
    return {"messages": [response]}


# Define the graph
builder = StateGraph(AgentState, input=InputState)

# Add nodes
builder.add_node("analyze_sentiment", analyze_sentiment)
builder.add_node("extract_context", extract_context)
builder.add_node("route_by_context", route_by_context)
builder.add_node("query_databases", query_databases)
builder.add_node("generate_tone", generate_tone)
builder.add_node("generate_deflection", generate_deflection)
builder.add_node("compose_response", compose_response)
builder.add_node("fact_check", fact_check)
builder.add_node("generate_final_output", generate_final_output)

# Add edges
builder.add_edge(START, ["analyze_sentiment", "extract_context"])
builder.add_edge("extract_context", "route_by_context")
builder.add_edge("route_by_context", "query_databases")
builder.add_edge("query_databases", check_data_found)
builder.add_edge("analyze_sentiment", ["generate_tone", "generate_deflection"])
builder.add_edge("generate_tone", "compose_response")
builder.add_edge("generate_deflection", "compose_response")
builder.add_edge("compose_response", "fact_check")
builder.add_edge("fact_check", "generate_final_output")
builder.add_edge("generate_final_output", END)

# Compile the graph
graph = builder.compile()
graph.name = "PoliticalAgentGraph"
