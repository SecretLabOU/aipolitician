from .trump import TrumpAgent
from .biden import BidenAgent

def get_agent(agent_name: str):
    agents = {
        "donald-trump": TrumpAgent(),
        "joe-biden": BidenAgent()
    }
    return agents.get(agent_name.lower().replace(" ", "-"))