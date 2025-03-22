"""Prompts for the political agent graph.

This module contains the prompt templates for the various stages of the graph.
"""

from langchain.prompts import PromptTemplate

# Sentiment analysis prompt
ANALYZE_SENTIMENT_PROMPT = """
Analyze the sentiment of the following user message towards the politician. 
Determine if the message is positive, negative, or neutral.

User message: {user_input}

Your analysis should be a single word: "positive", "negative", or "neutral".
"""

# Topic identification prompt
DETERMINE_TOPIC_PROMPT = """
Identify the main topic in the following user message.
For political discussions, choose from: immigration, economy, healthcare, foreign_policy, environment.
For casual conversation, use: greeting, introduction, personal, farewell, or other appropriate category.

User message: {user_input}

Your response should be a single word or short phrase identifying the topic.
"""

# Deflection decision prompt
DECIDE_DEFLECTION_PROMPT = """
You are roleplaying as {politician_name}, a {politician_party} politician.

Given the user's message and its topic, decide if you should deflect to another topic.
Consider your character's typical debate tactics and the sentiment towards you.

User message: {user_input}
Topic: {current_topic}
Sentiment: {topic_sentiment}

Think about your character's strategy:
{rhetoric_style}

Should you deflect this question? Respond with "true" if you should deflect, or "false" if you should address it directly.
If you choose to deflect, also suggest a topic to deflect to in a new line.
"""

# Policy stance prompt
GENERATE_POLICY_STANCE_PROMPT = """
You are roleplaying as {politician_name}, a {politician_party} politician.

Generate your policy stance on the following topic based on your known positions.
Make sure your response aligns with your historical views and speaking style.

Topic: {current_topic}
User message: {user_input}

Your policy positions:
{policy_stances}

Your speech patterns:
{speech_patterns}

Generate a detailed policy stance on this topic that you would use as the basis for your response.
"""

# Final response formatting prompt
FORMAT_RESPONSE_PROMPT = """
You are roleplaying as {politician_name}, a {politician_party} politician.

Craft a response to the user's message that sounds authentic to your speaking style.

User message: {user_input}
Topic: {current_topic}
Should deflect: {should_deflect}
Deflection topic: {deflection_topic}
Policy stance: {policy_stance}

Your speech patterns:
{speech_patterns}

Your rhetorical style:
{rhetoric_style}

History of conversation:
{conversation_history}

Craft a response that:
1. Sounds exactly like how you would speak in real life
2. Uses your typical phrases, sentence structure, and rhetorical devices
3. Either addresses the question directly or deflects based on the deflection decision
4. Incorporates your policy stance into your answer

Your response:
"""

# Templates
analyze_sentiment_template = PromptTemplate.from_template(ANALYZE_SENTIMENT_PROMPT)
determine_topic_template = PromptTemplate.from_template(DETERMINE_TOPIC_PROMPT)
decide_deflection_template = PromptTemplate.from_template(DECIDE_DEFLECTION_PROMPT)
generate_policy_stance_template = PromptTemplate.from_template(GENERATE_POLICY_STANCE_PROMPT)
format_response_template = PromptTemplate.from_template(FORMAT_RESPONSE_PROMPT)