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
import numpy as np
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

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
    formatted_name = name.replace(' ', '_')
    formatted_name_dash = name.replace(' ', '-')
    formatted_name_plus = name.replace(' ', '+')
    
    return [
        # Wikipedia and encyclopedias
        f"https://en.wikipedia.org/wiki/{formatted_name}",
        f"https://www.britannica.com/biography/{formatted_name_dash}",
        
        # News sources
        f"https://www.reuters.com/search/news?blob={name}",
        f"https://apnews.com/search?q={formatted_name_plus}",
        f"https://www.bbc.com/news/topics/{formatted_name_dash}",
        f"https://www.npr.org/search?query={formatted_name_plus}",
        f"https://www.cnn.com/search?q={formatted_name_plus}",
        f"https://www.foxnews.com/search-results/{formatted_name_plus}",
        
        # Government sources
        f"https://www.congress.gov/search?q=%7B%22source%22%3A%22members%22%2C%22search%22%3A%22{formatted_name_plus}%22%7D",
        f"https://www.govtrack.us/congress/members/find?q={formatted_name_plus}",
        
        # Speech archives
        f"https://www.c-span.org/search/?query={formatted_name_plus}",
        f"https://millercenter.org/the-presidency/presidential-speeches?field_president_target_id=All&keys={formatted_name_plus}",
        
        # Voting records
        f"https://www.govtrack.us/congress/votes/presidential-candidates",
        f"https://votesmart.org/candidate/key-votes/{formatted_name_dash}",
        
        # Fact-checking sites
        f"https://www.politifact.com/personalities/{formatted_name_dash}/",
        f"https://www.factcheck.org/?s={formatted_name_plus}",
    ]

# Direct extraction function using Ollama API
def extract_with_ollama(text, name, max_length=8000):
    """Extract structured information directly using Ollama API with robust error handling"""
    # Trim text to reasonable length
    text = text[:max_length]
    
    prompt = f"""
    Extract ONLY factual information about {name} from this text.
    
    Return a JSON object with THESE EXACT fields:
    {{
        "basic_info": {{
            "full_name": "Complete name including middle names",
            "date_of_birth": "YYYY-MM-DD format",
            "place_of_birth": "City, State/Province, Country",
            "nationality": "Country of citizenship",
            "political_affiliation": "Political party or affiliation",
            "education": ["Educational qualifications with institutions and years"],
            "family": ["Purely factual family information"]
        }},
        "career": {{
            "positions": ["All political positions held with dates"],
            "pre_political_career": ["Previous occupations before entering politics"],
            "committees": ["Committee memberships"]
        }},
        "policy_positions": {{
            "economy": ["Economic policy positions with direct quotes when available"],
            "foreign_policy": ["Foreign policy positions with direct quotes"],
            "healthcare": ["Healthcare policy positions with direct quotes"],
            "immigration": ["Immigration policy positions with direct quotes"],
            "environment": ["Environmental policy positions with direct quotes"],
            "social_issues": ["Social policy positions with direct quotes"],
            "other": ["Other policy positions not covered above"]
        }},
        "legislative_record": {{
            "sponsored_bills": ["Bills sponsored with dates"],
            "voting_record": ["Significant votes with dates"],
            "achievements": ["Legislative achievements"]
        }},
        "communications": {{
            "speeches": ["Notable speeches with dates and direct quotes"],
            "statements": ["Notable public statements with dates and direct quotes"],
            "publications": ["Books, articles or papers authored"]
        }},
        "campaigns": {{
            "elections": ["Electoral campaigns with years and results"],
            "platforms": ["Campaign platforms with direct quotes"],
            "fundraising": ["Campaign finance information"]
        }},
        "controversies": {{
            "legal_issues": ["Factual information about legal challenges or investigations"],
            "scandals": ["Factually reported controversies with no editorial judgment"]
        }},
        "timeline": ["Chronological key events with dates"]
    }}
    
    For any field where information is not available, use an empty string or empty array [].
    Do NOT include ANY subjective assessments, opinions, or conjectures.
    ONLY include verifiable facts and direct quotes from the source material.
    Return ONLY the JSON object, no additional text, explanations or formatting.
    
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
            },
            timeout=60  # Add timeout to prevent hanging
        )
        
        # Check if response is valid
        if response.status_code != 200:
            print(f"Error: Ollama API returned status code {response.status_code}")
            return None
            
        result = response.json()
        return result.get('response', None)
    except requests.exceptions.Timeout:
        print("Error: Ollama extraction request timed out")
        return None
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to Ollama API")
        return None
    except json.JSONDecodeError:
        print("Error: Received invalid JSON response from Ollama")
        return None
    except Exception as e:
        print(f"Error with Ollama extraction: {e}")
        return None

def get_article_text(url, selector=None, max_retries=2):
    """Fetch and extract the main content from a webpage with retries"""
    retries = 0
    
    while retries <= max_retries:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=20)
            
            # Check if request was successful
            if response.status_code != 200:
                print(f"Warning: Got status code {response.status_code} for {url}")
                retries += 1
                if retries <= max_retries:
                    print(f"Retrying ({retries}/{max_retries})...")
                    continue
                return None
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Different extraction based on domain
            if "wikipedia.org" in url:
                content_div = soup.find(id="mw-content-text")
                if content_div:
                    # Get paragraphs but exclude tables and references
                    paragraphs = content_div.find_all('p')[:25]
                    return "\n".join([p.get_text() for p in paragraphs])
            
            elif "britannica.com" in url:
                content_div = soup.find(class_="topic-content")
                if content_div:
                    paragraphs = content_div.find_all('p')[:15]
                    return "\n".join([p.get_text() for p in paragraphs])
            
            elif "congress.gov" in url:
                main_content = soup.find(id="main")
                if main_content:
                    paragraphs = main_content.find_all(['p', 'li', 'h2', 'h3'])[:20]
                    return "\n".join([p.get_text() for p in paragraphs])
                    
            elif "c-span.org" in url:
                main_content = soup.find("div", class_="video-content")
                if main_content:
                    paragraphs = main_content.find_all(['p', 'li'])[:20]
                    return "\n".join([p.get_text() for p in paragraphs])
                    
            elif "votesmart.org" in url:
                main_content = soup.find("div", class_="breakdown")
                if main_content:
                    paragraphs = main_content.find_all(['p', 'li', 'div', 'span'])[:30]
                    return "\n".join([p.get_text() for p in paragraphs])
                    
            elif "politifact.com" in url:
                main_content = soup.find("div", class_="m-statements")
                if main_content:
                    items = main_content.find_all(['div', 'p'])[:20]
                    return "\n".join([i.get_text() for i in items])
            
            elif selector:
                # Use custom selector if provided
                content = soup.select(selector)
                if content:
                    return "\n".join([p.get_text() for p in content])
            
            # Fallback: get all paragraphs and headings
            all_content = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li'])[:30]
            extracted_text = "\n".join([p.get_text() for p in all_content])
            
            # If we got meaningful content, return it
            if extracted_text and len(extracted_text) > 50:
                return extracted_text
            
            # If content is too short, try once more with different selectors
            if retries < max_retries:
                print(f"Extracted content too short ({len(extracted_text) if extracted_text else 0} chars), retrying with different selectors...")
                retries += 1
                # Try with different selectors on retry
                all_content = soup.find_all(['div', 'article', 'section', 'main'])[:10]
                extracted_text = "\n".join([p.get_text() for p in all_content])
                if extracted_text and len(extracted_text) > 50:
                    return extracted_text
            
            # If we still don't have good content, return what we have
            return extracted_text or ""
            
        except requests.exceptions.Timeout:
            print(f"Timeout fetching {url}")
            retries += 1
            if retries <= max_retries:
                print(f"Retrying ({retries}/{max_retries})...")
            else:
                return None
        except requests.exceptions.ConnectionError:
            print(f"Connection error fetching {url}")
            retries += 1
            if retries <= max_retries:
                print(f"Retrying ({retries}/{max_retries})...")
            else:
                return None
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            retries += 1
            if retries <= max_retries:
                print(f"Retrying ({retries}/{max_retries})...")
            else:
                return None
    
    return None

def generate_embedding(text, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Generate embedding vector for the text using the specified model with error handling"""
    try:
        # Ensure text is valid
        if not text or not isinstance(text, str):
            print("Warning: Invalid text for embedding, using fallback")
            text = "Unknown politician"
            
        print(f"Generating embedding using {model_name}...")
        model = SentenceTransformer(model_name)
        
        # Limit text length to prevent memory issues
        if len(text) > 10000:
            print(f"Text too long ({len(text)} chars), truncating for embedding")
            text = text[:10000]
            
        embedding = model.encode(text)
        print(f"✅ Generated embedding vector of dimension {len(embedding)}")
        return embedding.tolist()  # Convert to list for JSON serialization
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return [0.0] * 768  # Return zero vector as fallback

def extract_json_safely(text):
    """Safely extract JSON from text with multiple fallback strategies"""
    if not text:
        return None
        
    # Strategy 1: Direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print("Direct JSON parsing failed, trying regex extraction")
    
    # Strategy 2: Find JSON object with regex
    try:
        json_match = re.search(r'({[\s\S]*})', text)
        if json_match:
            return json.loads(json_match.group(1))
    except (json.JSONDecodeError, re.error):
        print("Regex JSON extraction failed, trying bracket balancing")
    
    # Strategy 3: Find balanced brackets
    try:
        bracket_count = 0
        start_idx = text.find('{')
        
        if start_idx == -1:
            return None
            
        for i in range(start_idx, len(text)):
            if text[i] == '{':
                bracket_count += 1
            elif text[i] == '}':
                bracket_count -= 1
                if bracket_count == 0:
                    json_str = text[start_idx:i+1]
                    return json.loads(json_str)
    except (json.JSONDecodeError, IndexError):
        print("Bracket balancing failed, returning None")
    
    return None

async def crawl_political_figure(name):
    urls = get_sources(name)
    
    # Initialize result collection with all required fields
    combined_data = {
        "id": str(uuid.uuid4()),
        "name": name,
        "basic_info": {
            "full_name": "",
            "date_of_birth": "",
            "place_of_birth": "",
            "nationality": "",
            "political_affiliation": "",
            "education": [],
            "family": []
        },
        "career": {
            "positions": [],
            "pre_political_career": [],
            "committees": []
        },
        "policy_positions": {
            "economy": [],
            "foreign_policy": [],
            "healthcare": [],
            "immigration": [],
            "environment": [],
            "social_issues": [],
            "other": []
        },
        "legislative_record": {
            "sponsored_bills": [],
            "voting_record": [],
            "achievements": []
        },
        "communications": {
            "speeches": [],
            "statements": [],
            "publications": []
        },
        "campaigns": {
            "elections": [],
            "platforms": [],
            "fundraising": []
        },
        "controversies": {
            "legal_issues": [],
            "scandals": []
        },
        "timeline": [],
        "embedding": []  # Will be populated later
    }
    
    # Process each source URL in parallel tasks
    try:
        tasks = []
        for url in urls:
            tasks.append(process_url(url, name, combined_data))
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)  # Add return_exceptions=True to prevent task failures from stopping execution
    except Exception as e:
        print(f"Error in parallel processing: {e}")
    
    # Generate embedding from all policy positions with safe handling
    try:
        # Construct a safe biography text for embedding
        biography_parts = [combined_data["name"]]
        
        # Safely add basic info strings
        for field in ["full_name", "political_affiliation"]:
            value = combined_data["basic_info"].get(field, "")
            if value and isinstance(value, str):
                biography_parts.append(value)
        
        # Safely add career positions
        if combined_data["career"]["positions"]:
            for pos in combined_data["career"]["positions"]:
                if pos and isinstance(pos, str):
                    biography_parts.append(pos)
                elif isinstance(pos, (list, dict)):
                    # Handle nested structures by converting to string
                    try:
                        pos_str = str(pos)
                        if pos_str and pos_str != "[]" and pos_str != "{}":
                            biography_parts.append(pos_str)
                    except:
                        pass
        
        # Safely add policy positions
        for policy_type, positions in combined_data["policy_positions"].items():
            if positions and isinstance(positions, list):
                for pos in positions:
                    if pos and isinstance(pos, str):
                        biography_parts.append(pos)
                    elif isinstance(pos, (list, dict)):
                        try:
                            pos_str = str(pos)
                            if pos_str and pos_str != "[]" and pos_str != "{}":
                                biography_parts.append(pos_str)
                        except:
                            pass
        
        # Join everything to create embedding text
        biography_text = " ".join(biography_parts)
        
        combined_data["embedding"] = generate_embedding(biography_text)
    except Exception as e:
        print(f"Error generating embedding: {e}")
        combined_data["embedding"] = [0.0] * 768  # Use fallback embedding
    
    # Clean up and validate date formats
    try:
        if combined_data["basic_info"]["date_of_birth"]:
            combined_data["basic_info"]["date_of_birth"] = standardize_date_format(
                combined_data["basic_info"]["date_of_birth"]
            )
    except Exception as e:
        print(f"Error standardizing date format: {e}")
    
    # Safely remove duplicate entries from lists
    for section in ["career", "policy_positions", "legislative_record", "communications", "campaigns", "controversies"]:
        for key in combined_data[section]:
            try:
                # Convert items to strings and remove duplicates
                safe_items = []
                for item in combined_data[section][key]:
                    try:
                        if item:  # Ensure item is not None or empty
                            safe_items.append(str(item))
                    except:
                        pass  # Skip items that can't be converted
                
                combined_data[section][key] = list(set(safe_items))
            except Exception as e:
                print(f"Error deduplicating {section}.{key}: {e}")
                combined_data[section][key] = []  # Reset to empty list on error
    
    # Safely deduplicate timeline
    try:
        safe_timeline = []
        for item in combined_data["timeline"]:
            try:
                if item:  # Ensure item is not None or empty
                    safe_timeline.append(str(item))
            except:
                pass  # Skip items that can't be converted
                
        combined_data["timeline"] = list(set(safe_timeline))
    except Exception as e:
        print(f"Error deduplicating timeline: {e}")
        combined_data["timeline"] = []  # Reset to empty list on error
    
    return combined_data

async def process_url(url, name, combined_data):
    """Process a single URL and update the combined data with comprehensive error handling"""
    try:
        print(f"Processing: {url}")
        
        # Get main content text
        article_text = get_article_text(url)
        if not article_text or len(article_text) < 100:  # Ensure we have meaningful content
            print(f"Failed to extract sufficient content from {url}")
            return
            
        print(f"Extracted {len(article_text)} characters of content")
        
        # Extract structured data using Ollama
        extracted_json_text = extract_with_ollama(article_text, name)
        if not extracted_json_text:
            print(f"Failed to extract structured data from {url}")
            return
            
        print(f"Got extraction result of length {len(extracted_json_text)}")
        
        # Try to parse the extraction result as JSON with enhanced safety
        data = extract_json_safely(extracted_json_text)
        
        if data:
            # Update combined data with deep merging
            merge_extracted_data(combined_data, data)
            print(f"Successfully extracted structured data from {url}")
        else:
            print(f"Could not parse JSON from the extraction result")
            
    except Exception as e:
        print(f"Error processing {url}: {e}")

def merge_extracted_data(combined_data, new_data):
    """Merge new extracted data into combined data with deduplication and error handling"""
    try:
        # Handle basic info (strings)
        if "basic_info" in new_data and isinstance(new_data["basic_info"], dict):
            for key, value in new_data["basic_info"].items():
                if key in combined_data["basic_info"]:
                    if isinstance(value, str) and value and not combined_data["basic_info"][key]:
                        combined_data["basic_info"][key] = value
                    elif isinstance(value, list) and value:
                        # Safely add list items
                        for item in value:
                            if item and item not in combined_data["basic_info"][key]:
                                try:
                                    combined_data["basic_info"][key].append(str(item))
                                except:
                                    # Skip items that can't be converted to string
                                    pass
    except Exception as e:
        print(f"Error merging basic info: {e}")
    
    # Handle nested dictionaries with list values
    for section in ["career", "policy_positions", "legislative_record", "communications", "campaigns", "controversies"]:
        try:
            if section in new_data and isinstance(new_data[section], dict):
                for key, value in new_data[section].items():
                    if key in combined_data[section]:
                        if isinstance(value, list):
                            # Safely add list items
                            for item in value:
                                if item and item not in combined_data[section][key]:
                                    try:
                                        # Safely convert item to string if needed
                                        if not isinstance(item, str):
                                            item = str(item)
                                        combined_data[section][key].append(item)
                                    except:
                                        # Skip items that can't be handled
                                        pass
        except Exception as e:
            print(f"Error merging {section}: {e}")
    
    # Handle timeline which is a simple list
    try:
        if "timeline" in new_data and isinstance(new_data["timeline"], list):
            for item in new_data["timeline"]:
                if item and item not in combined_data["timeline"]:
                    try:
                        # Safely convert item to string if needed
                        if not isinstance(item, str):
                            item = str(item)
                        combined_data["timeline"].append(item)
                    except:
                        # Skip items that can't be handled
                        pass
    except Exception as e:
        print(f"Error merging timeline: {e}")

def standardize_date_format(date_str):
    """Convert various date formats to YYYY-MM-DD with error handling"""
    if not date_str or not isinstance(date_str, str):
        return ""
        
    try:
        date_patterns = [
            # MM/DD/YYYY
            (r'(\d{1,2})/(\d{1,2})/(\d{4})', lambda m: f"{m.group(3)}-{m.group(1).zfill(2)}-{m.group(2).zfill(2)}"),
            # Month DD, YYYY
            (r'([A-Za-z]+) (\d{1,2}), (\d{4})', lambda m: convert_text_month(m.group(1), m.group(2), m.group(3))),
            # DD Month YYYY
            (r'(\d{1,2}) ([A-Za-z]+) (\d{4})', lambda m: convert_text_month(m.group(2), m.group(1), m.group(3))),
            # YYYY-MM-DD (already correct)
            (r'(\d{4})-(\d{1,2})-(\d{1,2})', lambda m: f"{m.group(1)}-{m.group(2).zfill(2)}-{m.group(3).zfill(2)}"),
        ]
        
        for pattern, formatter in date_patterns:
            match = re.search(pattern, date_str)
            if match:
                try:
                    return formatter(match)
                except:
                    # If formatter fails, continue to next pattern
                    continue
        
        return date_str  # Return original if no pattern matches
    except Exception as e:
        print(f"Error standardizing date format: {e}")
        return date_str  # Return original on exception

def convert_text_month(month_name, day, year):
    """Convert text month to numeric month"""
    month_mapping = {
        'january': '01', 'february': '02', 'march': '03', 'april': '04',
        'may': '05', 'june': '06', 'july': '07', 'august': '08',
        'september': '09', 'october': '10', 'november': '11', 'december': '12',
        'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04', 'jun': '06',
        'jul': '07', 'aug': '08', 'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
    }
    
    month_lower = month_name.lower()
    month_num = month_mapping.get(month_lower, '01')  # Default to 01 if not found
    
    return f"{year}-{month_num}-{day.zfill(2)}"

def save_to_json(data, name):
    """Save the data to a JSON file in the logs directory with error handling"""
    try:
        logs_dir = os.path.join("scraper", "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c if c.isalnum() or c == '_' else '_' for c in name)  # Sanitize filename
        filename = f"{safe_name}_{timestamp}.json"
        filepath = os.path.join(logs_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return filepath
    except Exception as e:
        print(f"Error saving data to JSON: {e}")
        # Try an alternative save method with a simpler path
        try:
            backup_path = f"politician_data_{timestamp}.json"
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return backup_path
        except:
            print("Failed to save data even with backup method")
            return None

def main(name, env_id="nat", gpu_count=1):
    try:
        # Set up GPU environment
        if not setup_gpu_environment(env_id, gpu_count):
            print("⚠️ GPU environment setup failed, continuing without GPU acceleration")
        
        print(f"Starting enhanced AI-powered extraction for {name} using Ollama/Llama3...")
        start_time = datetime.datetime.now()
        
        data = asyncio.run(crawl_political_figure(name))
        
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"Extraction completed in {duration:.2f} seconds")
        
        # Safely print summary of collected data
        try:
            print(f"\n--- DATA SUMMARY FOR {name} ---")
            basic_info_count = len([x for x in data['basic_info'].values() if x])
            print(f"Basic info: {basic_info_count} fields populated")
            positions_count = len(data['career'].get('positions', []))
            print(f"Positions: {positions_count} entries")
            
            policy_count = sum(len(data['policy_positions'].get(k, [])) for k in data['policy_positions'])
            print(f"Policy positions: {policy_count} across {len(data['policy_positions'])} categories")
            
            legislative_count = sum(len(data['legislative_record'].get(k, [])) for k in data['legislative_record'])
            print(f"Legislative record: {legislative_count} entries")
            
            comms_count = sum(len(data['communications'].get(k, [])) for k in data['communications'])
            print(f"Communications: {comms_count} entries")
            
            campaign_count = sum(len(data['campaigns'].get(k, [])) for k in data['campaigns'])
            print(f"Campaigns: {campaign_count} entries")
            
            timeline_count = len(data.get('timeline', []))
            print(f"Timeline: {timeline_count} events")
            
            embedding_dim = len(data.get('embedding', []))
            print(f"Embedding dimension: {embedding_dim}")
            print("----------------------------")
        except Exception as e:
            print(f"Error generating data summary: {e}")
        
        # Save data to JSON file
        filepath = save_to_json(data, name)
        if filepath:
            print(f"Data saved to: {filepath}")
        
        return data
    except Exception as e:
        print(f"Critical error in main function: {e}")
        # Create minimal data to return even on major failure
        return {
            "id": str(uuid.uuid4()),
            "name": name,
            "error": f"Failed to complete extraction: {str(e)}"
        }

if __name__ == "__main__":
    try:
        # Parse command line arguments
        env_id = "nat"  # Default environment ID
        gpu_count = 1   # Default GPU count
        
        # Allow overriding from command line
        if len(sys.argv) > 1:
            political_figure_name = sys.argv[1]
        else:
            political_figure_name = "Donald Trump"  # Default
            
        if len(sys.argv) > 2:
            env_id = sys.argv[2]
        
        if len(sys.argv) > 3:
            gpu_count = int(sys.argv[3])
        
        main(political_figure_name, env_id, gpu_count)
    except Exception as e:
        print(f"Fatal error in script execution: {e}")
        sys.exit(1)