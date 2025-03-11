import asyncio
import uuid
import json
import os
import datetime
import re
import requests
import subprocess
import sys
import atexit
from bs4 import BeautifulSoup

# GPU Environment Setup
def setup_gpu_environment(env_id="nat", gpu_count=1):
    """Setup genv environment and attach GPU"""
    print(f"Setting up GPU environment (id: {env_id}, GPUs: {gpu_count})...")
    
    try:
        # Activate the environment
        activate_cmd = f"genv activate --id {env_id}"
        subprocess.run(activate_cmd, shell=True, check=True)
        print(f"✅ Activated genv environment: {env_id}")
        
        # Attach GPUs
        attach_cmd = f"genv attach --count {gpu_count}"
        subprocess.run(attach_cmd, shell=True, check=True)
        print(f"✅ Attached {gpu_count} GPU(s) to environment")
        
        # Show GPU status
        subprocess.run("nvidia-smi", shell=True)
        
        # Register cleanup on exit
        atexit.register(cleanup_gpu_environment)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to setup GPU environment: {e}")
        return False
    except Exception as e:
        print(f"❌ Error setting up GPU: {e}")
        return False

def cleanup_gpu_environment():
    """Clean up by detaching GPUs"""
    try:
        print("\nCleaning up GPU environment...")
        # Detach all GPUs
        subprocess.run("genv detach --all", shell=True, check=True)
        print("✅ Successfully detached all GPUs")
    except Exception as e:
        print(f"⚠️ Error during GPU cleanup: {e}")

# Define target URLs
def get_sources(name):
    return [
        f"https://en.wikipedia.org/wiki/{name.replace(' ', '_')}",
        f"https://www.britannica.com/biography/{name.replace(' ', '-')}",
        f"https://www.reuters.com/search/news?blob={name}",
    ]

# Direct extraction function using Ollama API
def extract_with_ollama(text, name, max_length=4000):
    """Extract structured information directly using Ollama API"""
    # Trim text to reasonable length
    text = text[:max_length]
    
    prompt = f"""
    Extract key information about {name} from this text.
    
    Return a JSON object with ONLY these fields:
    {{
        "biography": "Brief biography focusing on early life and career",
        "date_of_birth": "MM/DD/YYYY format",
        "nationality": "Country of citizenship",
        "political_affiliation": "Political party or affiliation",
        "positions": ["List of political positions held"],
        "policies": ["List of notable policy positions"],
        "legislative_actions": ["List of notable legislative actions"]
    }}
    
    For any field where information is not available, use an empty string or empty list.
    Return ONLY the JSON object, no additional text or explanation.
    
    Text to analyze:
    {text}
    """
    
    try:
        print(f"Sending {len(text)} characters to Ollama for extraction...")
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
        print(f"Error with Ollama extraction: {e}")
        return None

def get_article_text(url, selector=None):
    """Fetch and extract the main content from a webpage"""
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Different extraction based on domain
        if "wikipedia.org" in url:
            content_div = soup.find(id="mw-content-text")
            if content_div:
                # Get the first few paragraphs
                paragraphs = content_div.find_all('p')[:10]
                return "\n".join([p.get_text() for p in paragraphs])
        
        elif "britannica.com" in url:
            content_div = soup.find(class_="topic-content")
            if content_div:
                paragraphs = content_div.find_all('p')[:8]
                return "\n".join([p.get_text() for p in paragraphs])
        
        elif selector:
            # Use custom selector if provided
            content = soup.select(selector)
            if content:
                return "\n".join([p.get_text() for p in content])
        
        # Fallback: get all paragraphs
        paragraphs = soup.find_all('p')[:15]
        return "\n".join([p.get_text() for p in paragraphs])
    
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

async def crawl_political_figure(name):
    urls = get_sources(name)
    
    # Initialize result collection
    combined_data = {
        "id": str(uuid.uuid4()),
        "name": name,
        "date_of_birth": "",
        "nationality": "",
        "political_affiliation": "",
        "biography": "",
        "positions": [],
        "policies": [],
        "legislative_actions": [],
        "campaigns": []
    }
    
    # Process each source URL
    for url in urls:
        try:
            print(f"Processing: {url}")
            
            # Get main content text
            article_text = get_article_text(url)
            if not article_text:
                print(f"Failed to extract content from {url}")
                continue
                
            print(f"Extracted {len(article_text)} characters of content")
            
            # Extract structured data using Ollama
            extracted_json_text = extract_with_ollama(article_text, name)
            if not extracted_json_text:
                print(f"Failed to extract structured data from {url}")
                continue
                
            print(f"Got extraction result of length {len(extracted_json_text)}")
            
            # Try to parse the extraction result as JSON
            try:
                # Remove any text before or after the JSON object
                json_match = re.search(r'({.*})', extracted_json_text, re.DOTALL)
                if json_match:
                    clean_json_text = json_match.group(1)
                    data = json.loads(clean_json_text)
                    
                    # Update combined data
                    for key, value in data.items():
                        if key in combined_data:
                            if isinstance(combined_data[key], list) and isinstance(value, list):
                                # Add unique items to list
                                combined_data[key].extend([item for item in value if item not in combined_data[key]])
                            elif isinstance(combined_data[key], str) and isinstance(value, str) and not combined_data[key]:
                                combined_data[key] = value
                                
                    print(f"Successfully extracted structured data from {url}")
                else:
                    print(f"Could not identify JSON object in the extraction result")
                    
            except json.JSONDecodeError as e:
                print(f"Failed to parse extraction result as JSON: {e}")
                
        except Exception as e:
            print(f"Error processing {url}: {e}")
    
    return combined_data

def save_to_json(data, name):
    """Save the data to a JSON file in the logs directory"""
    logs_dir = os.path.join("scraper", "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name.replace(' ', '_')}_{timestamp}.json"
    filepath = os.path.join(logs_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return filepath

def main(name, env_id="nat", gpu_count=1):
    # Set up GPU environment
    if not setup_gpu_environment(env_id, gpu_count):
        print("⚠️ GPU environment setup failed, continuing without GPU acceleration")
    
    print(f"Starting AI-powered extraction for {name} using Ollama/Llama3...")
    start_time = datetime.datetime.now()
    
    data = asyncio.run(crawl_political_figure(name))
    
    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"Extraction completed in {duration:.2f} seconds")
    
    # Print summary of collected data
    if data["biography"]:
        print(f"Biography: {data['biography'][:200]}...")
    else:
        print("No biography found.")
        
    print(f"Positions found: {len(data['positions'])}")
    print(f"Policies found: {len(data['policies'])}")
    print(f"Legislative actions: {len(data['legislative_actions'])}")
    
    # Save data to JSON file
    filepath = save_to_json(data, name)
    print(f"Data saved to: {filepath}")
    
    # No need to explicitly call cleanup_gpu_environment()
    # It will be called automatically when the script exits due to atexit.register()
    
    return data

if __name__ == "__main__":
    # Parse command line arguments
    env_id = "nat"  # Default environment ID
    gpu_count = 1   # Default GPU count
    
    # Allow overriding from command line
    if len(sys.argv) > 1:
        political_figure_name = sys.argv[1]
    else:
        political_figure_name = "Barack Obama"  # Default
        
    if len(sys.argv) > 2:
        env_id = sys.argv[2]
    
    if len(sys.argv) > 3:
        gpu_count = int(sys.argv[3])
    
    main(political_figure_name, env_id, gpu_count)