# Routing Agent
class ContextRoutingAgent:
    """
    Routes the extracted context to the appropriate database.

    Methods:
        route_context(context: str) -> str:
            Determines which database to query based on the context.
            Parameters:
                context (str): The extracted context.
            Returns:
                str: The name of the relevant database or None if no match is found.
    """
    def route_context(self, context):
        if "voting" in context:
            return "Voting DB"
        elif "policy" in context:
            return "Policy DB"
        elif "social" in context:
            return "Social DB"
        elif "bio" in context:
            return "Bio DB"
        else:
            return None