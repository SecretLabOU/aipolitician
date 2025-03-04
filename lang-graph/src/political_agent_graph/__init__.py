"""Political AI

Simple API for talking to AI politicians that simulate real politicians' views and speaking styles.
"""

from political_agent_graph.graph import graph
from political_agent_graph.state import AgentState, InputState
from political_agent_graph.mock_db import persona_manager, select_persona, get_available_personas
from political_agent_graph.config import set_model_for_task

__all__ = [
    "persona_manager",
    "select_persona",
    "get_available_personas",
    "run_conversation",
    "use_trump_model"
]

# Initialize the Trump model for specific tasks
def use_trump_model():
    """Configure the system to use the Trump fine-tuned model."""
    set_model_for_task("tone_generation", "local/trump_mistral")
    set_model_for_task("deflection", "local/trump_mistral")
    set_model_for_task("response_composition", "local/trump_mistral")
    set_model_for_task("final_output", "local/trump_mistral")
    set_model_for_task("multi_persona", "local/trump_mistral")
    set_model_for_task("default", "local/trump_mistral")

# Auto-initialize the Trump model on import
use_trump_model()

async def run_conversation(message: str, persona_id: str = None):
    """Talk to an AI politician.
    
    Simple function to ask a question to a politician and get their response.
    
    Args:
        message: Your question or statement
        persona_id: Optional politician ID to select (e.g., "bernie_sanders")
        
    Returns:
        The politician's response
    
    Example:
        ```python
        import asyncio
        from political_agent_graph import run_conversation, select_persona
        
        async def main():
            # Talk to Bernie
            select_persona("bernie_sanders")
            response = await run_conversation("What do you think about healthcare?")
            print(response)
        
        asyncio.run(main())
        ```
    """
    from langchain_core.messages import HumanMessage
    
    # Set persona if specified
    if persona_id:
        select_persona(persona_id)
    
    # Run the graph
    input_state = {"messages": [HumanMessage(content=message)]}
    result = await graph.ainvoke(input_state)
    
    # Return the response
    if result.get("messages"):
        return result["messages"][0].content
    return "No response generated."
