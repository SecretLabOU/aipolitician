"""Integration between existing chat scripts and the Political Agent Graph.

This module provides the interface between the existing chat scripts and the
Political Agent Graph system, allowing the chat scripts to use the more sophisticated
multi-component architecture when needed.
"""

import os
import sys
import asyncio
from typing import Optional, Dict, Any
from pathlib import Path

# Ensure the src directory is in the path
root_dir = Path(__file__).parent.parent.parent.absolute()
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Import the Political Agent Graph components
try:
    from political_agent_graph.graph import run_conversation
    from political_agent_graph import set_active_persona
    HAS_GRAPH = True
except ImportError as e:
    print(f"Political Agent Graph not available: {e}")
    HAS_GRAPH = False

def generate_response_with_graph(
    prompt: str, 
    persona_id: str,
    use_rag: bool = False
) -> str:
    """
    Generate a response using the Graph architecture
    
    Args:
        prompt: The user's input prompt
        persona_id: ID of the political persona (e.g., "donald_trump")
        use_rag: Whether to use RAG (currently handled within the graph)
        
    Returns:
        The generated response text
    """
    if not HAS_GRAPH:
        return f"Graph system not available. Using fallback response for {persona_id}."
    
    # Set the active persona
    set_active_persona(persona_id)
    
    # Use asyncio to run the conversation
    try:
        # Run the conversation
        response = asyncio.run(run_conversation(prompt))
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error generating response with graph: {str(e)}"

def generate_policy_stance(state: ConversationState) -> ConversationState:
    """Generate the politician's policy stance."""
    model = get_model_for_task("generate_policy_stance")
    
    # Get active persona
    persona = persona_manager.get_active_persona()
    
    # Determine which topic to use
    topic = state.deflection_topic if state.should_deflect else state.current_topic
    
    # Add RAG integration here
    rag_context = ""
    if HAS_RAG:
        try:
            # Get factual information from the RAG system
            rag_context = integrate_with_chat(
                state.user_input, 
                persona["name"]
            )
        except Exception as e:
            print(f"Error using RAG: {e}")
    
    # Get the relevant policy stance from the persona data
    policy_data = persona.get("policy_stances", {})
    relevant_policy = policy_data.get(topic, {})
    
    prompt = generate_policy_stance_template.format(
        politician_name=persona["name"],
        politician_party=persona["party"],
        current_topic=topic,
        user_input=state.user_input,
        factual_context=rag_context,  # Add RAG context to the prompt
        policy_stances=json.dumps(relevant_policy, indent=2) if relevant_policy else "No specific stance on this topic.",
        speech_patterns=json.dumps(persona["speech_patterns"], indent=2)
    )
    
    # Get policy stance from model
    policy_stance = model.invoke(prompt).strip()
    
    # Update state
    state.policy_stance = policy_stance
    return state
