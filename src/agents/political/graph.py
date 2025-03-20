from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from src.models.chat.chat_trump import generate_response as trump_generate
from src.models.chat.chat_biden import generate_response as biden_generate

async def retrieve_context(state: PoliticalAgentState, config: RunnableConfig):
    """Retrieve relevant context using RAG."""
    from src.data.db.utils.rag_utils import integrate_with_chat
    
    persona_name = "Donald Trump" if state.persona == "trump" else "Joe Biden"
    state.retrieved_context = integrate_with_chat(state.query, persona_name)
    return state

async def generate_response(state: PoliticalAgentState, config: RunnableConfig):
    """Generate response using the appropriate model."""
    prompt = f"{state.retrieved_context}\n\nUser Question: {state.query}"
    
    if state.persona == "trump":
        state.final_response = trump_generate(
            prompt=prompt,
            model=None,  # Will use global model
            tokenizer=None,  # Will use global tokenizer
            use_rag=False  # We've already retrieved context
        )
    else:
        state.final_response = biden_generate(
            prompt=prompt,
            model=None,
            tokenizer=None,
            use_rag=False
        )
    
    return state

def create_political_graph():
    """Create the political agent graph."""
    
    graph = StateGraph(PoliticalAgentState)
    
    # Add nodes
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("generate_response", generate_response)
    
    # Define edges
    graph.set_entry_point("retrieve_context")
    graph.add_edge("retrieve_context", "generate_response")
    graph.add_edge("generate_response", END)
    
    return graph.compile()
