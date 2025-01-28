from swarms import Agent, HuggingFace
from dotenv import load_dotenv
import os
load_dotenv()

huggingface_token = os.getenv("HUGGINGFACE_API_KEY")

# Initialize the Hugging Face model with a Llama model
llm = HuggingFace(
    model_name="meta-llama/Llama-2-7b-chat-hf",  # Llama model
    temperature=0.5,  # Adjust creativity
    max_length=400,   # Maximum tokens to generate
    huggingface_token=huggingface_token  # Required for gated models
)

# Create the agent with Llama
agent = Agent(
    agent_name="Trump",
    llm=llm,  # Use the Hugging Face model
    max_loops="auto",
    interactive=True,
    streaming_on=True,
)

# Run the agent
agent.run("Who is Donald Trump")