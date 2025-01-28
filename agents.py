from swarms import Agent, HuggingFace
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get Hugging Face API token
huggingface_token = os.getenv("HUGGINGFACE_API_KEY")

# Initialize a smaller Hugging Face model for testing
llm = HuggingFace(
    model_name="distilbert-base-uncased",  # Smaller, lightweight model
    temperature=0.5,  # Adjust creativity
    max_length=100,   # Smaller token limit for testing
    huggingface_token=huggingface_token  # Optional for non-gated models
)

# Create the agent
agent = Agent(
    agent_name="TestAgent",
    llm=llm,  # Use the Hugging Face model
    max_loops="auto",
    interactive=True,
    streaming_on=True,
)

# Run the agent
agent.run("Who is Donald Trump?")
