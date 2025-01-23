from ..SentimentAgent.agent import SentimentAgent
from ..ContextAgent.agent import ContextAgent
from ..RoutingAgent.agent import RoutingAgent
from ..ToneAgent.agent import ToneAgent
from ..DeflectionAgent.agent import DeflectionAgent
from ..FactCheckingAgent.agent import FactCheckingAgent

# Workflow Manager
class WorkflowManager:
    """
    Manages the workflow by orchestrating interactions between different agents.

    Attributes:
        sentiment_agent (SentimentAgent): Instance of the SentimentAgent class.
        context_agent (ContextAgent): Instance of the ContextAgent class.
        routing_agent (RoutingAgent): Instance of the RoutingAgent class.
        tone_agent (ToneAgent): Instance of the ToneAgent class.
        deflection_agent (DeflectionAgent): Instance of the DeflectionAgent class.
        fact_checking_agent (FactCheckingAgent): Instance of the FactCheckingAgent class.

    Methods:
        multi_agent_workflow(user_input: str) -> str:
            Executes the workflow for processing user input and generating a response.
            Parameters:
                user_input (str): The text input from the user.
            Returns:
                str: Verified response generated by the agents.
    """
    def __init__(self):
        self.sentiment_agent = SentimentAgent()
        self.context_agent = ContextAgent()
        self.routing_agent = RoutingAgent()
        self.tone_agent = ToneAgent()
        self.deflection_agent = DeflectionAgent()
        self.fact_checking_agent = FactCheckingAgent()

    def multi_agent_workflow(self, user_input):
        sentiment_score = self.sentiment_agent.analyze_sentiment(user_input)
        context = self.context_agent.extract_context(user_input)
        relevant_db = self.routing_agent.route_context(context)

        if relevant_db:
            aggregated_data = f"Fetched data from {relevant_db} based on {context}"
            tone_adjusted_data = self.tone_agent.adjust_tone_with_persona(aggregated_data, sentiment_score)
            draft_response = f"Composed response: {tone_adjusted_data}"
        else:
            draft_response = self.deflection_agent.generate_deflection(context, sentiment_score)

        verified_response = self.fact_checking_agent.fact_check_response(draft_response)
        return verified_response