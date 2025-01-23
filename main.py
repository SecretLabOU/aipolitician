from Agents.WorkflowManager.agent import WorkflowManager
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
    print("Running Workflow Manager...")
    user_input = "Tell me about Donald Trump's policies on trade."
    workflow_manager = WorkflowManager()
    final_response = workflow_manager.multi_agent_workflow(user_input)
    print(final_response)

if __name__ == "__main__":
    main()