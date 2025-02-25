"""System prompts for the political agent graph.

This module defines the system prompts used by the various agents in the graph.
Each prompt is designed to guide the agent in performing its specific task.
"""

SENTIMENT_PROMPT = """Analyze the sentiment of the user's input.
Focus on emotional tone, attitude, and underlying feelings.
Return a brief description of the sentiment."""

CONTEXT_PROMPT = """Extract the main context and topic from the user's input.
Focus on identifying key subjects, themes, and information needs.
Return a brief description of the context."""

ROUTING_PROMPT = """Based on the extracted context, determine which databases should be queried.
Available databases: voting, bio, social, policy
Return a list of relevant database names."""

TONE_PROMPT = """Generate an appropriate tone for the response based on:
1. The user's sentiment
2. The aggregated data
3. The persona style

Return a brief description of the appropriate tone to use."""

DEFLECTION_PROMPT = """Generate an appropriate deflection based on:
1. The user's sentiment
2. The extracted context
3. The persona style

Return a diplomatic deflection that acknowledges the query without providing specifics."""

RESPONSE_PROMPT = """Compose a response using:
1. The determined tone
2. The aggregated data or deflection
3. The chat history context

Generate a natural, contextually appropriate response."""

FACT_CHECK_PROMPT = """Verify the accuracy of the draft response against:
1. The factual knowledge base
2. The aggregated data

Ensure all statements are supported by the available data."""

FINAL_OUTPUT_PROMPT = """Generate the final response ensuring:
1. It maintains the appropriate tone
2. It incorporates fact-checked information
3. It follows the persona style
4. It is contextually appropriate

Create a clear, concise, and well-structured response."""
