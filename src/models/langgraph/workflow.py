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
from src.models.langgraph.agents.context_agent import extract_context
from src.models.langgraph.agents.sentiment_agent import analyze_sentiment
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
    """Extract context with tracing."""
    if state.get("trace", False):
        print("\n🔎 TRACE: Context Agent - Starting")
        print("=====================================")
        print(f"Input: \"{state['user_input']}\"")
        print("-------------------------------------")
    
    result = extract_context(state)
    
    if state.get("trace", False):
        print("\n🔎 TRACE: Context Agent - Results")
        print("=====================================")
        print(f"Main Topics: {result.get('main_topics', 'None')}")
        print(f"Policy Areas: {result.get('policy_areas', 'None')}")
        print(f"Knowledge Retrieved: {'Yes' if result.get('has_knowledge', False) else 'No'}")
        print(f"Context Length: {len(result.get('context', ''))} characters")
        print("Context Preview: " + result.get('context', 'None')[:100] + "..." if len(result.get('context', '')) > 100 else result.get('context', 'None'))
        print("-------------------------------------")
    
    return result

def trace_sentiment_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze sentiment with tracing."""
    if state.get("trace", False):
        print("\n🔎 TRACE: Sentiment Agent - Starting")
        print("=====================================")
        print(f"Analyzing: \"{state['user_input']}\"")
        print("-------------------------------------")
    
    result = analyze_sentiment(state)
    
    if state.get("trace", False):
        print("\n🔎 TRACE: Sentiment Agent - Results")
        print("=====================================")
        print(f"Sentiment Score: {result.get('sentiment_score', 0):.2f} / 1.0")
        print(f"Sentiment Category: {result.get('sentiment_category', 'unknown')}")
        
        if 'emotion_scores' in result and result['emotion_scores']:
            print("\nEmotion Breakdown:")
            for emotion, score in result['emotion_scores'].items():
                print(f"  - {emotion.capitalize()}: {score:.2f}")
        
        print(f"\nDeflection Decision: {'Yes' if result.get('should_deflect', False) else 'No'}")
        if result.get('should_deflect', False):
            print(f"Deflection Reason: {result.get('deflection_reason', 'Negative sentiment detected')}")
        print("-------------------------------------")
    
    return result

def trace_response_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate response with tracing."""
    if state.get("trace", False):
        print("\n🔎 TRACE: Response Agent - Starting")
        print("=====================================")
        print(f"Politician: {state.get('politician_identity', 'unknown')}")
        print(f"Deflection Mode: {'Yes' if state.get('should_deflect', False) else 'No'}")
        print(f"Context Available: {'Yes' if state.get('context', '') else 'No'}")
        print("-------------------------------------")
    
    result = generate_response(state)
    
    if state.get("trace", False):
        print("\n🔎 TRACE: Response Agent - Results")
        print("=====================================")
        print(f"Response Generated: {len(result['response'])} characters")
        print("Response Preview: " + result['response'][:50] + "..." if len(result['response']) > 50 else result['response'])
        print("\n📝 Complete response will be displayed after all processing completes.")
        print("-------------------------------------")
    
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