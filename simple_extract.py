import requests
import json
from bs4 import BeautifulSoup

def get_wiki_content(name):
    """Get the first section of a Wikipedia page"""
    url = f"https://en.wikipedia.org/wiki/{name.replace(' ', '_')}"
    print(f"Fetching content from: {url}")
    
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Get the first paragraph which typically contains biographical info
        main_content = soup.find(id="mw-content-text")
        paragraphs = main_content.find_all('p')
        
        # Combine the first few paragraphs
        text = "\n".join([p.get_text() for p in paragraphs[:5]])
        return text
    except Exception as e:
        print(f"Error fetching Wikipedia: {e}")
        return None

def extract_info_with_ollama(text, name):
    """Extract information using Ollama API"""
    prompt = f"""
    Extract information about {name} from this Wikipedia text:
    
    1. Write a brief biography (1-2 sentences)
    2. When was this person born? (date format MM/DD/YYYY if available)
    3. What is their political party?
    
    Return ONLY JSON in this format:
    {{
      "biography": "...",
      "birth_date": "...", 
      "party": "..."
    }}
    
    Text to analyze:
    {text}
    """
    
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'llama3',
                'prompt': prompt,
                'stream': False
            }
        )
        
        result = response.json()
        return result['response']
    except Exception as e:
        print(f"Error with Ollama API: {e}")
        return None

def main():
    name = "Barack Obama"
    print(f"=== SIMPLE EXTRACTION TEST FOR {name} ===")
    
    # Get content
    content = get_wiki_content(name)
    if not content:
        print("Failed to get Wikipedia content.")
        return
    
    print(f"Got {len(content)} characters of content.")
    print("\nFirst 200 characters:")
    print(content[:200])
    
    # Extract information
    print("\nExtracting information with Ollama...")
    extraction = extract_info_with_ollama(content, name)
    if extraction:
        print("\nExtracted information:")
        print(extraction)
    else:
        print("\nFailed to extract information with Ollama.")

if __name__ == "__main__":
    main() 