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
        history_str = "\n".join([f"User: {msg[0]}\nTrump: {msg[1]}" for msg in history])
        prompt = f"The following is a conversation with Donald Trump. {self.personality_traits}\n\n"
        if history:
            prompt += f"{history_str}\n"
        prompt += f"User: {user_input}\nTrump:"
        return prompt
