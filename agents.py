from swarms import Agent, HuggingFace
from dotenv import load_dotenv
import os
import torch

# Load environment variables
load_dotenv()

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("CUDA is available!")
    print(f"Using device: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Using CPU.")

# Get Hugging Face API token
huggingface_token = os.getenv("HUGGINGFACE_API_KEY")

# Initialize a smaller Hugging Face model for testing
llm = HuggingFace(
    model_name="distilbert-base-uncased",  # Smaller, lightweight model
    temperature=0.5,  # Adjust creativity
    max_length=100,   # Smaller token limit for testing
    huggingface_token=huggingface_token,  # Optional for non-gated models
    device=device  # Use GPU if available
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
