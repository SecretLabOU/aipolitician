from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Retrieve the API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("API key not found! Ensure OPENAI_API_KEY is set in the .env file.")

# Placeholder for the main logic
def main():
    print(f"Successfully loaded API key")

if __name__ == "__main__":
    main()
