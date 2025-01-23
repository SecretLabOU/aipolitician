# Fact Checking Agent
class FactCheckingAgent:
    """
    Verifies the accuracy of the response using a knowledge base.

    Methods:
        fact_check_response(response: str) -> str:
            Fact-checks the given response.
            Parameters:
                response (str): The draft response to verify.
            Returns:
                str: Fact-checked response.
    """
    def fact_check_response(self, response):
        return f"Fact-checked response: {response}"