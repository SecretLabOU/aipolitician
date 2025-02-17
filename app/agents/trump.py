from .base import BaseAgent

class TrumpAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.personality_traits = (
            "Speak confidently and use hyperbole. "
            "Focus on themes like America First, economic success, "
            "and criticism of opponents. Use phrases like 'fake news' "
            "and 'make America great again'."
        )
    
    def format_prompt(self, user_input: str, history: list) -> str:
        history_str = "\n".join([f"User: {msg[0]}\nAgent: {msg[1]}" for msg in history])
        return f"""
        {self.personality_traits}
        {history_str}
        User: {user_input}
        Agent: [Trump Style Response] 
        """