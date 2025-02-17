from .base import BaseAgent

class BidenAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.personality_traits = (
            "Speak with empathy and focus on unity. "
            "Emphasize working across the aisle, middle-class values, "
            "and rebuilding America. Use folksy expressions."
        )
    
    def format_prompt(self, user_input: str, history: list) -> str:
        history_str = "\n".join([f"User: {msg[0]}\nAgent: {msg[1]}" for msg in history])
        return f"""
        {self.personality_traits}
        {history_str}
        User: {user_input}
        Agent: [Biden Style Response] 
        """