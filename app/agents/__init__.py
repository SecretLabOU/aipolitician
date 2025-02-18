from .trump import TrumpAgent

def get_agent(agent_name: str):
    agents = {
        "donald-trump": TrumpAgent()
    }
    return agents.get(agent_name.lower().replace(" ", "-"))
