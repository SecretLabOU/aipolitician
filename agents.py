from swarms import Agent
from swarm_models import HuggingFaceLLM
from dotenv import load_dotenv
import os
import torch

# Load environment variables
load_dotenv()

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    print("CUDA is available!")
    print(f"Using device: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Using CPU.")

# Initialize the Hugging Face model
llm = HuggingFaceLLM(
    model_id="distilbert-base-uncased",  # Smaller, lightweight model
    device=device,                      # Use GPU if available
    max_length=100                      # Maximum token limit for generation
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
