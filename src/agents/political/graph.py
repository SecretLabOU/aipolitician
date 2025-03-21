from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from src.models.chat.chat_trump import generate_response as trump_generate
from src.models.chat.chat_biden import generate_response as biden_generate
from .state import PoliticalAgentState

async def retrieve_context(state: PoliticalAgentState, config: RunnableConfig):
    """Retrieve relevant context using RAG."""
    try:
        from src.data.db.utils.rag_utils import integrate_with_chat
        persona_name = "Donald Trump" if state.persona == "trump" else "Joe Biden"
        state.retrieved_context = integrate_with_chat(state.query, persona_name)
    except ImportError:
        # Fallback behavior when RAG is not available
        state.retrieved_context = f"Context not available - RAG system not configured"
    return {"retrieved_context": state.retrieved_context}

async def generate_response(state: dict, config: RunnableConfig):
    """Generate response using the appropriate model."""
    query = state.get("query", "")
    persona = state.get("persona", "")
    retrieved_context = state.get("retrieved_context", "")
    
    prompt = f"{retrieved_context}\n\nUser Question: {query}"
    
    if persona == "trump":
        response = trump_generate(
            prompt=prompt,
            use_rag=False  # We've already retrieved context
        )
    else:
        response = biden_generate(
            prompt=prompt,
            use_rag=False  # We've already retrieved context
        )
    return {"final_response": response}

def create_political_graph() -> StateGraph:
    """Create the political agent workflow graph."""
    # Initialize the graph
    workflow = StateGraph(PoliticalAgentState)
    
    # Add nodes
    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("generate_response", generate_response)
    
    # Define the edges
    workflow.set_entry_point("retrieve_context")
    workflow.add_edge("retrieve_context", "generate_response")
    workflow.add_edge("generate_response", END)
    
    # Define the output structure
    workflow.set_finish_point("generate_response")
    
    return workflow.compile()
