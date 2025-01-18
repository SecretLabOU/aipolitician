from dotenv import load_dotenv
import os
from agents import agent

# Load environment variables
load_dotenv()

# Retrieve the OpenAI API key 
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("API key not found! Ensure OPENAI_API_KEY is set in the .env file.")

# Main function to run the agent
def main():
    print("Running Stock-Analysis-Agent...")
    # Execute the agent with a predefined query
    agent.run("What is the current market trend for tech stocks?")

if __name__ == "__main__":
    main()
