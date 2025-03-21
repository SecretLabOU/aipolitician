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
    return state

async def generate_response(state: PoliticalAgentState, config: RunnableConfig):
    """Generate response using the appropriate model."""
    prompt = f"{state.retrieved_context}\n\nUser Question: {state.query}"
    
    if state.persona == "trump":
        state.final_response = trump_generate(
            prompt=prompt,
            use_rag=False  # We've already retrieved context
        )
    else:
        state.final_response = biden_generate(
            prompt=prompt,
            use_rag=False  # We've already retrieved context
        )
    return state

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
    
    return workflow.compile()
