from typing import Optional
from .graph import create_political_graph
from .state import PoliticalAgentState

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
        state = PoliticalAgentState(
            query=query,
            persona=self.persona
        )
        
        # Run the graph
        final_state = await self.graph.ainvoke(state)
        
        return final_state.final_response
