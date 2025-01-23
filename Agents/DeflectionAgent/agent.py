# Deflection Agent
class DeflectionAgent:
    """
    Generates deflective responses when no relevant data is found.

    Methods:
        generate_deflection(context: str, sentiment_score: float) -> str:
            Creates a deflective response based on the context and sentiment score.
            Parameters:
                context (str): The extracted context.
                sentiment_score (float): Sentiment polarity score.
            Returns:
                str: Deflective response.
    """
    def generate_deflection(self, context, sentiment_score):
        return f"Deflective response based on {context} with sentiment {sentiment_score}"
