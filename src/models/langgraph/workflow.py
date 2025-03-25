#!/usr/bin/env python3
"""
LangGraph workflow for the AI Politician system.
This module defines the main workflow that connects all the agents.
"""
import sys
from pathlib import Path
from typing import Dict, Any, TypedDict, Annotated, Literal, List
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

# Add the project root to the Python path
root_dir = Path(__file__).parent.parent.parent.parent.absolute()
sys.path.insert(0, str(root_dir))

from src.models.langgraph.config import PoliticianIdentity
from src.models.langgraph.agents.context_agent import process_context
from src.models.langgraph.agents.sentiment_agent import process_sentiment
from src.models.langgraph.agents.response_agent import generate_response

# Define input/output schemas
class PoliticianInput(BaseModel):
    """Input schema for the AI Politician workflow."""
    user_input: str = Field(..., description="User's input/question")
    politician_identity: Literal["biden", "trump"] = Field(..., 
                                                          description="Identity of the politician to impersonate")
    use_rag: bool = Field(default=True, description="Whether to use RAG for knowledge retrieval")
    trace: bool = Field(default=False, description="Whether to output trace information")

class PoliticianOutput(BaseModel):
    """Output schema for the AI Politician workflow."""
    response: str = Field(..., description="Generated response from the politician")
    sentiment_analysis: Dict[str, Any] = Field(..., description="Analysis of user input sentiment")
    should_deflect: bool = Field(..., description="Whether the politician needed to deflect")
    has_knowledge: bool = Field(..., description="Whether relevant knowledge was found")

# State type for the workflow
class WorkflowState(TypedDict):
    user_input: str
    politician_identity: str
    use_rag: bool
    trace: bool
    context: str
    has_knowledge: bool
    sentiment_analysis: Dict[str, Any]
    should_deflect: bool
    response: str

# Wrap agent functions to add tracing
def trace_context_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Process context with tracing."""
    if state.get("trace", False):
        print("\n✅ CHECKPOINT: Context Agent - Extracting topics and retrieving knowledge...")
    
    result = process_context(state)
    
    if state.get("trace", False):
        print(f"✅ CHECKPOINT: Context Extracted: {result['context'].split('Knowledge Base Context')[0].strip()}")
        print(f"✅ CHECKPOINT: Knowledge Found: {'Yes' if result['has_knowledge'] else 'No'}")
    
    return result

def trace_sentiment_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Process sentiment with tracing."""
    if state.get("trace", False):
        print("\n✅ CHECKPOINT: Sentiment Agent - Analyzing sentiment...")
    
    result = process_sentiment(state)
    
    if state.get("trace", False):
        sentiment = result["sentiment_analysis"]
        print(f"✅ CHECKPOINT: Sentiment Score: {sentiment.get('sentiment_score', 0):.2f}")
        print(f"✅ CHECKPOINT: Sentiment Category: {sentiment.get('sentiment_category', 'unknown')}")
        print(f"✅ CHECKPOINT: Deflection Needed: {'Yes' if result['should_deflect'] else 'No'}")
    
    return result

def trace_response_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate response with tracing."""
    if state.get("trace", False):
        print("\n✅ CHECKPOINT: Response Agent - Generating response...")
    
    result = generate_response(state)
    
    if state.get("trace", False):
        print(f"✅ CHECKPOINT: Response Generated ({len(result['response'])} chars)")
        print("\n📝 Generated response will be displayed below after all processing completes.\n")
    
    return result

def create_politician_graph() -> StateGraph:
    """Create the LangGraph workflow for the AI Politician."""
    # Initialize the state graph with the appropriate state type
    workflow = StateGraph(WorkflowState)
    
    # Add nodes for each agent
    workflow.add_node("context_agent", trace_context_agent)
    workflow.add_node("sentiment_agent", trace_sentiment_agent)
    workflow.add_node("response_agent", trace_response_agent)
    
    # Define the edges (flow) of the graph
    # Start -> Context Agent
    workflow.set_entry_point("context_agent")
    
    # Context Agent -> Sentiment Agent
    workflow.add_edge("context_agent", "sentiment_agent")
    
    # Sentiment Agent -> Response Agent
    workflow.add_edge("sentiment_agent", "response_agent")
    
    # Response Agent -> End
    workflow.add_edge("response_agent", END)
    
    return workflow

def process_user_input(input_data: PoliticianInput) -> PoliticianOutput:
    """
    Process user input through the AI Politician workflow.
    
    Args:
        input_data: User input and configuration
        
    Returns:
        PoliticianOutput: The politician's response and metadata
    """
    # Create the graph
    graph = create_politician_graph()
    
    # Convert to runnable
    politician_chain = graph.compile()
    
    # Create initial state
    initial_state: WorkflowState = {
        "user_input": input_data.user_input,
        "politician_identity": input_data.politician_identity,
        "use_rag": input_data.use_rag,
        "trace": input_data.trace,
        "context": "",
        "has_knowledge": False,
        "sentiment_analysis": {},
        "should_deflect": False,
        "response": ""
    }
    
    # Run the workflow
    result = politician_chain.invoke(initial_state)
    
    # Return formatted output
    return PoliticianOutput(
        response=result["response"],
        sentiment_analysis=result["sentiment_analysis"],
        should_deflect=result["should_deflect"],
        has_knowledge=result["has_knowledge"]
    ) 