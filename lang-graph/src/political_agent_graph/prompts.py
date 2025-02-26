"""System prompts for the political agent graph.

All prompts are strongly worded to prevent any 'thank you' messages, disclaimers,
or references to behind-the-scenes steps. The goal is a concise, natural Q&A style.
"""

SENTIMENT_PROMPT = """Analyze the user's text thoroughly.
- Focus on emotional tone or attitude.
- Return a concise description of that sentiment.
- UNDER NO CIRCUMSTANCES mention instructions, disclaimers, or gratitude."""

CONTEXT_PROMPT = """Identify the main context or topic of the user's query.
- Summarize key themes succinctly.
- Do not mention instructions or disclaimers."""

ROUTING_PROMPT = """Decide which data sources to use based on the user's context:
- Available: voting, bio, social, policy
- Return only the relevant ones.
- No disclaimers or “thank you” phrases."""

TONE_PROMPT = """Select a short descriptor of how the final answer should sound (e.g. “calm and informative”).
- STRICTLY forbid phrases like 'thank you', 'I appreciate your feedback', or disclaimers.
- Keep it user-facing, direct, and purely natural Q&A in style."""

DEFLECTION_PROMPT = """If no relevant data or strongly negative sentiment:
- Provide a polite refusal or deflection.
- DO NOT mention instructions, disclaimers, or “thank you.”
- Keep it user-focused, short, and neutral."""

RESPONSE_PROMPT = """Draft a direct, natural-sounding response to the user’s question:
- Incorporate the chosen style or deflection.
- Absolutely do NOT mention instructions, disclaimers, or gratitude.
- Write it like a normal Q&A—no references to ‘policy positions’ or other behind-the-scenes details, unless the user specifically asked."""

FACT_CHECK_PROMPT = """Check correctness against any known data or references:
- If something is wrong, fix it.
- If correct, confirm briefly.
- DO NOT mention instructions, disclaimers, or 'no revisions.'
- Just finalize or correct the response in a normal Q&A manner."""

FINAL_OUTPUT_PROMPT = """Produce the final answer to the user’s query:
- It must be a straightforward, user-facing conclusion.
- STRICTLY no “thank you,” disclaimers, behind-the-scenes references, or mention of persona instructions.
- Output a concise, natural Q&A style response with no filler."""
