"""Prompts for the political agent graph.

Concise, focused prompt templates for the LangGraph implementation.
"""

from langchain.prompts import PromptTemplate

# Sentiment analysis prompt
ANALYZE_SENTIMENT_PROMPT = """
Analyze the sentiment of this message toward a politician: 
{user_input}

Respond with only: "positive", "negative", or "neutral".
"""

# Topic identification prompt
DETERMINE_TOPIC_PROMPT = """
Identify the main political topic in this message:
{user_input}

Respond with a single term like: immigration, economy, healthcare, foreign_policy, etc.
"""

# Context retrieval prompt
RETRIEVE_CONTEXT_TEMPLATE = """
Summarize the key facts from this information that help answer:
Question: {question}

Information:
{context}

Provide only relevant factual information.
"""

# Deflection decision prompt
DECIDE_DEFLECTION_PROMPT = """
You are {politician_name}, a {politician_party} politician.

Message: {user_input}
Topic: {current_topic}
Sentiment: {topic_sentiment}
Context: {context}

Strategy:
{rhetoric_style}

Should you deflect? Respond with:
true/false
[deflection topic if true]
"""

# Policy stance prompt
GENERATE_POLICY_STANCE_PROMPT = """
As {politician_name} ({politician_party}), generate your policy stance on:
Topic: {current_topic}
Message: {user_input}

Your policy positions:
{policy_stances}

Your speaking style:
{speech_patterns}

Additional context:
{context}

Generate a detailed policy stance in your authentic voice.
"""

# Fact check prompt
FACT_CHECK_TEMPLATE = """
Fact-check this politician's stance against the retrieved information:

Stance:
{policy_stance}

Facts:
{retrieved_context}

Identify any factual inconsistencies. If consistent, reply "CONSISTENT WITH FACTS."
"""

# Final response formatting prompt
FORMAT_RESPONSE_PROMPT = """
As {politician_name} ({politician_party}), craft a response to:
Message: {user_input}
Topic: {current_topic}
Deflect: {should_deflect}
Deflection topic: {deflection_topic}
Policy stance: {policy_stance}
{fact_check_info}

Your speech style:
{speech_patterns}

Your rhetoric:
{rhetoric_style}

Previous conversation:
{conversation_history}

Craft a response that sounds exactly like you would speak in real life.
"""

# Templates
analyze_sentiment_template = PromptTemplate.from_template(ANALYZE_SENTIMENT_PROMPT)
determine_topic_template = PromptTemplate.from_template(DETERMINE_TOPIC_PROMPT)
retrieve_context_template = PromptTemplate.from_template(RETRIEVE_CONTEXT_TEMPLATE)
decide_deflection_template = PromptTemplate.from_template(DECIDE_DEFLECTION_PROMPT)
generate_policy_stance_template = PromptTemplate.from_template(GENERATE_POLICY_STANCE_PROMPT)
fact_check_template = PromptTemplate.from_template(FACT_CHECK_TEMPLATE)
format_response_template = PromptTemplate.from_template(FORMAT_RESPONSE_PROMPT)