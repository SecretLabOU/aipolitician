from typing import Dict, Optional, List, TypedDict, Annotated, Sequence
from langgraph.graph import Graph, StateGraph
from transformers import pipeline
import uuid
from pydantic import BaseModel

class AgentState(TypedDict):
    messages: List[Dict[str, str]]
    current_speaker: str
    session_id: str
    context: Dict[str, any]
    next_step: str

class PoliticalAgent:
    def __init__(self, name: str, personality_traits: Dict[str, float]):
        """
        Initialize a political agent with a name and personality traits.
        
        Args:
            name: Name of the political figure
            personality_traits: Dictionary of personality traits and their strengths (0-1)
        """
        self.name = name
        self.personality_traits = personality_traits
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1  # CPU by default, will be updated if GPU is available
        )
        
        # Initialize state graph
        self.workflow = self._create_workflow()
        
    def _create_workflow(self) -> Graph:
        """Create the conversation workflow graph."""
        # Create state graph
        workflow = StateGraph(AgentState)
        
        # Define conditional edges
        def should_continue(state: AgentState) -> str:
            return state["next_step"]
        
        # Add nodes
        workflow.add_node("process_input", self._process_input)
        workflow.add_node("analyze_sentiment", self._analyze_sentiment_node)
        workflow.add_node("generate_response", self._generate_response_node)
        workflow.add_node("format_response", self._format_response_node)
        
        # Define edges with conditional routing
        workflow.add_edge("process_input", should_continue)
        workflow.add_edge("analyze_sentiment", should_continue)
        workflow.add_edge("generate_response", should_continue)
        workflow.add_edge("format_response", should_continue)
        
        # Set entry point
        workflow.set_entry_point("process_input")
        
        # Set exit point
        workflow.set_finish_point("format_response")
        
        # Compile graph
        return workflow.compile()
        
    def _process_input(self, state: AgentState) -> AgentState:
        """Process user input and update state."""
        if "session_id" not in state:
            state["session_id"] = str(uuid.uuid4())
        state["current_speaker"] = "user"
        state["next_step"] = "analyze_sentiment"
        return state

    def _analyze_sentiment_node(self, state: AgentState) -> AgentState:
        """Analyze sentiment of the latest message."""
        latest_message = state["messages"][-1]["content"]
        result = self.sentiment_analyzer(latest_message)[0]
        state["context"]["sentiment"] = {
            "label": result["label"],
            "score": result["score"]
        }
        state["next_step"] = "generate_response"
        return state
        
    def _generate_response_node(self, state: AgentState) -> AgentState:
        """Generate response based on conversation state."""
        # To be implemented by specific agent classes
        state["current_speaker"] = self.name
        state["next_step"] = "format_response"
        return state

    def _format_response_node(self, state: AgentState) -> AgentState:
        """Format the response according to personality."""
        # To be implemented by specific agent classes
        state["next_step"] = "end"
        return state
        
    async def generate_response(
        self,
        message: str,
        session_id: Optional[str] = None
    ) -> Dict:
        """
        Generate a response to the user's message using the workflow graph.
        
        Args:
            message: User's input message
            session_id: Optional session ID for continuing a conversation
            
        Returns:
            Dictionary containing response, session_id, and sentiment analysis
        """
        # Initialize or update state
        state: AgentState = {
            "messages": [{"role": "user", "content": message}],
            "current_speaker": "user",
            "session_id": session_id or str(uuid.uuid4()),
            "next_step": "process_input",
            "context": {
                "agent_name": self.name,
                "personality_traits": self.personality_traits
            }
        }
        
        try:
            # Execute workflow
            final_state = self.workflow.invoke(state)
            
            return {
                "response": final_state["messages"][-1]["content"],
                "session_id": final_state["session_id"],
                "sentiment": final_state["context"].get("sentiment", {}),
                "context": {
                    "agent_name": self.name,
                    "personality_traits": str(self.personality_traits)
                }
            }
        except Exception as e:
            print(f"Error in workflow execution: {str(e)}")
            raise
        
    def set_gpu_device(self, device_id: int):
        """Set GPU device for the agent's models."""
        self.sentiment_analyzer.device = device_id
        # Additional GPU settings can be configured here
