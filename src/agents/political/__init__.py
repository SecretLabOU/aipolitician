from typing import Optional
from .state import PoliticalAgentState
from .graph import create_political_graph

class PoliticalAgent:
    def __init__(self, persona: str):
        """Initialize political agent.
        
        Args:
            persona: Either "trump" or "biden"
        """
        self.persona = persona
        self.graph = create_political_graph()
    
    async def generate_response(self, query: str) -> str:
        """Generate a response to the given query."""
        
        # Initialize state
        initial_state = PoliticalAgentState(
            query=query,
            persona=self.persona
        )
        
        # Convert state to dict for graph processing
        state_dict = initial_state.dict()
        
        # Run the graph
        final_state = await self.graph.ainvoke(state_dict)
        
        # Convert back to state object
        final_state_obj = PoliticalAgentState.from_dict(final_state)
        
        return final_state_obj.final_response or "No response generated"
