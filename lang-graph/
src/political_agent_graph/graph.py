from shared.utils import load_chat_model


async def select_persona(state: AgentState, config: RunnableConfig) -> dict[str, Union[str, dict]]:
    """Select the persona to use for this conversation.
    
    This is a new node that implements persona selection at the start of the graph.
    It either uses a specified persona_id or detects the appropriate persona from context.
    """
    # Load available personas
    personas = load_personas()
    
    # If persona_id is already set in state, use that
    if state.persona_id:
        persona_id = state.persona_id
    else:
        # Default to first persona if none is specified
        persona_id = list(personas.keys())[0]
        
        # If first message, try to detect persona from user input
        if len(state.chat_memory) <= 1 and state.messages:
            model = load_chat_model("anthropic/claude-3-haiku-20240307")
            personas_list = ", ".join(personas.keys())
            messages = [
                {"role": "system", "content": f"The user is talking to a political agent. Determine which of the following personas they're most likely addressing based on their message: {personas_list}. Respond with just the name of the persona, nothing else."},
                {"role": "human", "content": state.messages[-1].content},
            ]
            response = await model.ainvoke(messages)
            detected_persona = response.content.strip()
            
            # Use detected persona if valid
            if detected_persona in personas:
                persona_id = detected_persona
    
    # Get detailed persona data
    persona_data = get_persona_data(persona_id)
    
    return {
        "persona_id": persona_id,
        "persona_data": persona_data
    }


async def analyze_sentiment(state: AgentState, config: RunnableConfig) -> dict[str, str]:
    # Use the selected persona data instead of generic persona style
    persona_data = state.persona_data
    
    model = load_chat_model("anthropic/claude-3-haiku-20240307")
    messages = [
        {"role": "system", "content": TONE_PROMPT},
        {"role": "human", "content": f"Sentiment: {state.sentiment}\nData: {state.aggregated_data}\nPersona ID: {state.persona_id}\nSpeech Pattern: {persona_data.get('speech_pattern', '')}\nRhetorical Style: {persona_data.get('rhetorical_style', '')}\nMemory: {state.persona_memory}"},
    ]
    # Use the selected persona data instead of generic persona style
    persona_data = state.persona_data
    
    model = load_chat_model("anthropic/claude-3-haiku-20240307")
    messages = [
        {"role": "system", "content": DEFLECTION_PROMPT},
        {"role": "human", "content": f"Sentiment: {state.sentiment}\nContext: {state.context}\nPersona ID: {state.persona_id}\nSpeech Pattern: {persona_data.get('speech_pattern', '')}\nRhetorical Style: {persona_data.get('rhetorical_style', '')}\nCatchphrases: {persona_data.get('catchphrases', '')}\nMemory: {state.persona_memory}"},
    ]
async def compose_response(state: AgentState, config: RunnableConfig) -> dict[str, str]:
    """Compose draft response.
    
    This implements the "Response Composer" node in the flowchart,
    which consults the Chat Memory to incorporate conversation history.
    """
    # Use the chat memory from state instead of separate DB
    chat_history = state.chat_memory
    persona_data = state.persona_data
    
    model = load_chat_model("anthropic/claude-3-haiku-20240307")
    content = f"""Tone: {state.tone}
Data/Deflection: {state.aggregated_data or state.deflection}
Chat History: {chat_history}
Persona ID: {state.persona_id}
Biography: {persona_data.get('biography', '')}
Policy Stances: {persona_data.get('policy_stances', '')}
Speech Pattern: {persona_data.get('speech_pattern', '')}
Catchphrases: {persona_data.get('catchphrases', '')}
Memory: {state.persona_memory}"""
    messages = [
        {"role": "system", "content": FINAL_OUTPUT_PROMPT},
        {
            "role": "human",
            "content": f"""Tone: {state.tone}
Persona ID: {state.persona_id}
Speech Pattern: {state.persona_data.get('speech_pattern', '')}
Rhetorical Style: {state.persona_data.get('rhetorical_style', '')}
Verified Response: {state.verified_response}""",
        },
    ]
    return {"messages": [response]}


async def update_memory(state: AgentState, config: RunnableConfig) -> dict[str, str]:
    """Update the memory for consistency across conversation turns.
    
    This implements a new node for maintaining persona-specific memory
    and consistent references to the persona's background and talking points.
    """
    # Generate a memory summary from the conversation
    model = load_chat_model("anthropic/claude-3-haiku-20240307")
    
    # Extract policy positions, commitments, key points mentioned
    persona_data = state.persona_data
    previous_memory = state.persona_memory
    
    messages = [
        {"role": "system", "content": """You are an expert memory system for a politician AI. 
Extract and update the key discussion points, policy positions, and commitments made in this 
conversation that should be remembered for consistency. Focus on:
1. Policy positions stated
2. Commitments or promises made
3. Personal anecdotes or stories mentioned
4. Emotional responses to specific topics
5. Recurring themes or talking points

Keep the existing memory where appropriate, but add or update new information.
Format your response as a structured but concise summary."""},
        {"role": "human", "content": f"Previous Memory: {previous_memory}\n\nLatest User Question: {state.messages[-1].content}\n\nResponse Given: {state.verified_response}\n\nPersona: {state.persona_id}\nPolicy Stances: {persona_data.get('policy_stances', '')}"},
    ]
    response = await model.ainvoke(messages)
    
    return {"persona_memory": response.content, "chat_memory": state.chat_memory + f"\nUser: {state.messages[-1].content}\nAI ({state.persona_id}): {state.verified_response}"}


# Define the graph
# Add nodes corresponding to the flowchart
builder.add_node("select_persona", select_persona)        # Persona Selection Node (new)
builder.add_node("analyze_sentiment", analyze_sentiment)  # Sentiment Agent
builder.add_node("extract_context", extract_context)      # Context Agent
builder.add_node("route_by_context", route_by_context)    # Routing Agent
builder.add_node("query_databases", query_databases)      # Database Queries
builder.add_node("generate_tone", generate_tone)          # Tone Agent
builder.add_node("generate_deflection", generate_deflection)  # Deflection Agent
builder.add_node("compose_response", compose_response)    # Response Composer
builder.add_node("fact_check", fact_check)                # Fact Checking Agent
builder.add_node("generate_final_output", generate_final_output)  # Final Output
builder.add_node("update_memory", update_memory)          # Memory Update Node (new)
# Add edges to match the flowchart flow
# 0. User Input -> Persona Selection
builder.add_edge(START, "select_persona")

# 1. Persona Selection -> Parallel flow to Sentiment & Context
builder.add_edge("select_persona", "analyze_sentiment")
builder.add_edge("select_persona", "extract_context")

# 2. Context -> Routing -> Database Queries
builder.add_edge("extract_context", "route_by_context")
builder.add_edge("route_by
