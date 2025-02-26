"""Main implementation of the political agent graph.

This module implements the flowchart as a LangGraph, defining the nodes and edges
that process user input through sentiment analysis, context extraction, database queries,
and response generation.

The graph follows this flow:
1. User input is processed in parallel by Sentiment and Context agents
2. Context is used by the Routing agent to select appropriate databases
3. Selected databases (Voting, Bio, Social, Policy) are queried
4. Based on whether data is found:
   - If data is found: Generate appropriate tone using Persona DB
   - If no data is found: Generate a deflection
5. Response is composed using tone/deflection and Chat Memory
6. Draft response is fact-checked against Factual Knowledge Base
7. Final output is generated and returned to the user
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
    """Analyze sentiment of user input.
    
    This implements the "Sentiment Agent" node in the flowchart,
    which processes the user input to determine sentiment.
    """
    model = load_chat_model("anthropic/claude-3-haiku-20240307")
    messages = [
        {"role": "system", "content": SENTIMENT_PROMPT},
        {"role": "human", "content": state.messages[-1].content},
    ]
    response = await model.ainvoke(messages)
    return {"sentiment": response.content}


async def extract_context(state: AgentState, config: RunnableConfig) -> dict[str, str]:
    """Extract context from user input.
    
    This implements the "Context Agent" node in the flowchart,
    which processes the user input to extract relevant context.
    """
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
    """Route to appropriate databases based on context.
    
    This implements the "Select Database(s)?" decision point in the flowchart.
    The function determines which databases (Voting, Bio, Social, Policy)
    should be queried based on the extracted context.
    """
    model = load_chat_model("anthropic/claude-3-haiku-20240307")
    model = model.with_structured_output(RouterResponse)
    messages = [
        {"role": "system", "content": ROUTING_PROMPT},
        {"role": "human", "content": state.context},
    ]
    response = cast(RouterResponse, await model.ainvoke(messages))
    return {"selected_databases": response["selected_databases"]}


async def query_databases(state: AgentState, config: RunnableConfig) -> dict[str, str]:
    """Query selected databases and aggregate results.
    
    This implements the database query flows in the flowchart,
    where each selected database (Voting DB, Bio DB, Social DB, Policy DB)
    is queried individually based on the routing decision.
    """
    results = {}
    aggregated = []
    
    # Query each selected database individually
    for db in state.selected_databases:
        if db == "voting":
            # Query Voting DB
            results["voting_data"] = DB_REGISTRY["voting"](state.context)
            aggregated.append(results["voting_data"])
        elif db == "bio":
            # Query Bio DB
            results["bio_data"] = DB_REGISTRY["bio"](state.context)
            aggregated.append(results["bio_data"])
        elif db == "social":
            # Query Social DB
            results["social_data"] = DB_REGISTRY["social"](state.context)
            aggregated.append(results["social_data"])
        elif db == "policy":
            # Query Policy DB
            results["policy_data"] = DB_REGISTRY["policy"](state.context)
            aggregated.append(results["policy_data"])
    
    # Aggregate the results from all queried databases
    results["aggregated_data"] = " | ".join(aggregated)
    return results


def check_data_found(state: AgentState) -> Literal["generate_tone", "generate_deflection"]:
    """Check if data was found and route accordingly.
    
    This is the "Is Data Found?" decision point in the flowchart.
    If data is found, route to tone generation.
    If no data is found, route to deflection generation.
    """
    # Check if aggregated data exists and is not empty
    if state.aggregated_data.strip():
        return "generate_tone"  # Yes, data found
    else:
        return "generate_deflection"  # No data found


async def generate_tone(state: AgentState, config: RunnableConfig) -> dict[str, str]:
    """Generate appropriate tone for response.
    
    This implements the "Tone Agent" node in the flowchart,
    which checks the Persona DB to determine the appropriate tone.
    """
    # Get persona style from Persona DB
    persona_style = DB_REGISTRY["persona"]("")
    
    model = load_chat_model("anthropic/claude-3-haiku-20240307")
    messages = [
        {"role": "system", "content": TONE_PROMPT},
        {"role": "human", "content": f"Sentiment: {state.sentiment}\nData: {state.aggregated_data}\nPersona: {persona_style}"},
    ]
    response = await model.ainvoke(messages)
    return {"tone": response.content, "persona_style": persona_style}


async def generate_deflection(state: AgentState, config: RunnableConfig) -> dict[str, str]:
    """Generate appropriate deflection.
    
    This implements the "Deflection Agent" node in the flowchart,
    which generates a diplomatic deflection when no relevant data is found.
    """
    # Get persona style from Persona DB
    persona_style = DB_REGISTRY["persona"]("")
    
    model = load_chat_model("anthropic/claude-3-haiku-20240307")
    messages = [
        {"role": "system", "content": DEFLECTION_PROMPT},
        {"role": "human", "content": f"Sentiment: {state.sentiment}\nContext: {state.context}\nPersona: {persona_style}"},
    ]
    response = await model.ainvoke(messages)
    return {"deflection": response.content, "persona_style": persona_style}


async def compose_response(state: AgentState, config: RunnableConfig) -> dict[str, str]:
    """Compose draft response.
    
    This implements the "Response Composer" node in the flowchart,
    which consults the Chat Memory DB to incorporate conversation history.
    """
    # Get chat history from Chat Memory DB
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
    """Fact check the draft response.
    
    This implements the "Fact Checking Agent" node in the flowchart,
    which verifies the draft response against the Factual Knowledge Base.
    """
    # Get factual knowledge from the knowledge base
    facts = DB_REGISTRY["factual_kb"]("")
    
    # Verify the draft response against the factual knowledge base
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
    """Generate final output.
    
    This implements the "Final Output" node in the flowchart,
    which produces the final response to the user.
    """
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

# Add nodes corresponding to the flowchart
builder.add_node("analyze_sentiment", analyze_sentiment)  # Sentiment Agent
builder.add_node("extract_context", extract_context)      # Context Agent
builder.add_node("route_by_context", route_by_context)    # Routing Agent
builder.add_node("query_databases", query_databases)      # Database Queries
builder.add_node("generate_tone", generate_tone)          # Tone Agent
builder.add_node("generate_deflection", generate_deflection)  # Deflection Agent
builder.add_node("compose_response", compose_response)    # Response Composer
builder.add_node("fact_check", fact_check)                # Fact Checking Agent
builder.add_node("generate_final_output", generate_final_output)  # Final Output

# Add edges to match the flowchart flow
# 1. Parallel flow: User Input -> Sentiment & Context
builder.add_edge(START, "analyze_sentiment")
builder.add_edge(START, "extract_context")

# 2. Context -> Routing -> Database Queries
builder.add_edge("extract_context", "route_by_context")
builder.add_edge("route_by_context", "query_databases")

# 3. Data Found Decision Point
builder.add_conditional_edges("query_databases", check_data_found)

# 4. Sentiment flows to both Tone and Deflection
def route_sentiment(state: AgentState) -> Literal["generate_tone", "generate_deflection"]:
    if "negative" in state.sentiment.lower():
        return "generate_deflection"
    else:
        return "generate_tone"

builder.add_conditional_edges("analyze_sentiment", route_sentiment)

# 5. Tone/Deflection -> Response Composer
builder.add_edge("generate_tone", "compose_response")
builder.add_edge("generate_deflection", "compose_response")

# 6. Response Composer -> Fact Check -> Final Output
builder.add_edge("compose_response", "fact_check")
builder.add_edge("fact_check", "generate_final_output")
builder.add_edge("generate_final_output", END)

# Compile the graph
graph = builder.compile()
graph.name = "PoliticalAgentGraph"
