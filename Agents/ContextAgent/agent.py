# Context Agent
class ContextAgent:
    """
    Extracts context or topic from user input.

    Methods:
        extract_context(user_text: str) -> str:
            Processes the input text to identify the context or topic.
            Parameters:
                user_text (str): The input text from the user.
            Returns:
                str: Extracted context based on the input.
    """
    def extract_context(self, user_text):
        return "Extracted context based on input: " + user_text