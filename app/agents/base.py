from typing import Dict, Optional, List, TypedDict, Annotated, Sequence
from langgraph.graph import Graph, StateGraph
import uuid
from pydantic import BaseModel

from ..utils.model_manager import ModelManager
from ..utils.session_manager import SessionManager
from ..utils.exceptions import (
    ModelLoadError,
    ModelGenerationError,
    SessionError,
    InvalidRequestError
)

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
        
        try:
            # Initialize sentiment analyzer
            model_dict = ModelManager.get_model(
                "distilbert-base-uncased-finetuned-sst-2-english",
                "sentiment-analysis"
            )
            self.sentiment_analyzer = model_dict["generator"]
        except Exception as e:
            raise ModelLoadError(f"Failed to load sentiment analyzer: {str(e)}")
        
        # Initialize state graph
        self.workflow = self._create_workflow()
        
    def _create_workflow(self) -> Graph:
        """Create the conversation workflow graph."""
        try:
            # Create state graph
            workflow = StateGraph(AgentState)
            
            # Define conditional edges
            def should_continue(state: AgentState) -> str:
                return state["next_step"]
            
            # Add router node first
            workflow.add_node("router", should_continue)
            
            # Add processing nodes
            workflow.add_node("process_input", self._process_input)
            workflow.add_node("analyze_sentiment", self._analyze_sentiment_node)
            workflow.add_node("generate_response", self._generate_response_node)
            workflow.add_node("format_response", self._format_response_node)
            
            # Define edges with conditional routing
            workflow.add_edge("process_input", "router")
            workflow.add_edge("analyze_sentiment", "router")
            workflow.add_edge("generate_response", "router")
            workflow.add_edge("format_response", "router")
            
            # Set entry point
            workflow.set_entry_point("process_input")
            
            # Set exit point
            workflow.set_finish_point("format_response")
            
            # Compile graph
            return workflow.compile()
            
        except Exception as e:
            raise InvalidRequestError(f"Failed to create workflow: {str(e)}")
        
    def _process_input(self, state: AgentState) -> AgentState:
        """Process user input and update state."""
        try:
            if "session_id" not in state:
                state["session_id"] = str(uuid.uuid4())
                SessionManager.create_session(state["session_id"], self.name)
            else:
                # Load existing session
                session = SessionManager.get_session(state["session_id"])
                if session:
                    state["messages"] = session["messages"]
            
            state["current_speaker"] = "user"
            state["next_step"] = "analyze_sentiment"
            return state
            
        except Exception as e:
            raise SessionError(f"Failed to process input: {str(e)}")

    def _analyze_sentiment_node(self, state: AgentState) -> AgentState:
        """Analyze sentiment of the latest message."""
        try:
            latest_message = state["messages"][-1]["content"]
            result = self.sentiment_analyzer(latest_message)[0]
            state["context"]["sentiment"] = {
                "label": result["label"],
                "score": result["score"]
            }
            state["next_step"] = "generate_response"
            return state
            
        except Exception as e:
            raise ModelGenerationError(f"Failed to analyze sentiment: {str(e)}")
        
    def _generate_response_node(self, state: AgentState) -> AgentState:
        """Generate response based on conversation state."""
        try:
            # To be implemented by specific agent classes
            state["current_speaker"] = self.name
            state["next_step"] = "format_response"
            return state
            
        except Exception as e:
            raise ModelGenerationError(f"Failed to generate response: {str(e)}")

    def _format_response_node(self, state: AgentState) -> AgentState:
        """Format the response according to personality."""
        try:
            # To be implemented by specific agent classes
            state["next_step"] = "end"
            
            # Update session with new messages
            SessionManager.update_session(
                state["session_id"],
                state["messages"],
                state["context"]
            )
            
            return state
            
        except Exception as e:
            raise ModelGenerationError(f"Failed to format response: {str(e)}")
        
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
        try:
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
            
            # Load existing session if provided
            if session_id:
                session = SessionManager.get_session(session_id)
                if session:
                    state["messages"] = session["messages"] + state["messages"]
            
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
            raise ModelGenerationError(f"Failed to generate response: {str(e)}")
        
    def set_gpu_device(self, device_id: int):
        """Set GPU device for the agent's models."""
        try:
            ModelManager.set_gpu_device(
                "distilbert-base-uncased-finetuned-sst-2-english",
                device_id
            )
        except Exception as e:
            print(f"Warning: Failed to set GPU device: {str(e)}")
