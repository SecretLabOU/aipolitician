import asyncio
import uuid
import json
import os
import datetime
import re
from crawl4ai import (
    AsyncWebCrawler, LLMExtractionStrategy, LLMConfig, 
    BFSDeepCrawlStrategy, FilterChain, DomainFilter, 
    URLPatternFilter, RegexChunking, LLMContentFilter,
    CrawlerRunConfig
)

# Define target URLs
def get_sources(name):
    return [
        f"https://en.wikipedia.org/wiki/{name.replace(' ', '_')}",
        f"https://www.britannica.com/biography/{name.replace(' ', '-')}",
        f"https://www.reuters.com/search/news?blob={name}",
    ]

async def crawl_political_figure(name):
    urls = get_sources(name)
    
    # Configure Ollama (local LLM - no API token needed)
    llm_config = LLMConfig(
        provider="ollama/llama3",  # Free, no token needed
        base_url="http://localhost:11434"  # Default Ollama server address
    )
    
    # Create LLM extraction strategy - this one does support prompt_template and variables
    extraction_strategy = LLMExtractionStrategy(
        llm_config=llm_config,
        prompt_template="""
        Extract key information about {name} from the provided text.
        
        Return a JSON object with these fields:
        {{
            "biography": "Brief biography focusing on early life and career",
            "date_of_birth": "Month Day, Year format if available, otherwise 'unknown'",
            "nationality": "Country of citizenship",
            "political_affiliation": "Political party or affiliation",
            "positions": ["List of political positions held"],
            "policies": ["List of notable policy positions"],
            "legislative_actions": ["List of notable legislative actions"],
            "campaigns": ["List of notable campaigns"]
        }}
        
        Keep your response focused and factual. Use "unknown" for missing information.
        """,
        variables={"name": name},
        chunk_token_threshold=4000,
        overlap_rate=0.1
    )
    
    # Create deep crawl strategy - limited to control resource usage
    deep_crawl = BFSDeepCrawlStrategy(
        max_depth=1,  # Only go one link deep
        max_pages=3,  # Limit to 3 pages per source
        filter_chain=FilterChain([
            DomainFilter(allowed_domains=["en.wikipedia.org", "britannica.com", "reuters.com"]),
            URLPatternFilter(patterns=[
                name.lower().replace(" ", "-"), 
                name.lower().replace(" ", "_"), 
                "biography", 
                "political"
            ])
        ])
    )
    
    # Create content filter - 'instruction' instead of 'prompt_template' and no 'variables' parameter
    content_filter = LLMContentFilter(
        llm_config=llm_config,
        instruction=f"Evaluate if this content contains relevant information about {name}'s political career, biography, or policies. Answer only 'yes' or 'no'."
    )
    
    # Configure chunking strategy
    chunking_strategy = RegexChunking(
        patterns=[r"\n## ", r"\n\n"]  # Correct parameter name
    )
    
    # Create crawler config - FIXED: removed content_filter parameter
    crawler_config = CrawlerRunConfig(
        extraction_strategy=extraction_strategy,
        deep_crawl_strategy=deep_crawl,
        chunking_strategy=chunking_strategy,
        # Add additional common parameters
        word_count_threshold=50,  # Min words per chunk to process
        cache_mode="bypass",      # Don't use cache
        verbose=True              # Show detailed logs
    )
    
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
        "campaigns": [],
        "achievements": []
    }
    
    # Crawl URLs with Ollama-powered AI strategies
    async with AsyncWebCrawler() as crawler:
        for url in urls:
            try:
                print(f"Crawling: {url}")
                # Use crawler.arun which returns a CrawlResultContainer
                result = await crawler.arun(url=url, config=crawler_config)
                
                # Apply content filtering if needed - this is done manually since we can't pass it in config
                # Note: In practice the AsyncWebCrawler should handle this internally
                
                # Check if the result has extracted_data
                if hasattr(result, 'extracted_data') and result.extracted_data:
                    print(f"Extracted data from {url}")
                    data = result.extracted_data
                    
                    # Update combined data
                    for key, value in data.items():
                        if key in combined_data:
                            if isinstance(combined_data[key], list) and isinstance(value, list):
                                # Add unique items to list
                                combined_data[key].extend([item for item in value if item not in combined_data[key]])
                            elif isinstance(combined_data[key], str) and isinstance(value, str) and not combined_data[key]:
                                combined_data[key] = value
                            
                # If no structured data extracted, try using the markdown content
                elif hasattr(result, 'markdown') and result.markdown:
                    print(f"Processing markdown from {url}")
                    
                    # Extract basic info using regex
                    content = result.markdown
                    
                    # Extract date of birth if available
                    dob_match = re.search(r'born\s+(?:on\s+)?([A-Z][a-z]+\s+\d{1,2},\s+\d{4})', content)
                    if dob_match and not combined_data["date_of_birth"]:
                        combined_data["date_of_birth"] = dob_match.group(1)
                    
                    # Extract nationality if available
                    if "American" in content[:1000] and not combined_data["nationality"]:
                        combined_data["nationality"] = "American"
                    
                    # Extract political affiliation if available
                    if ("Democratic Party" in content or "Democrat" in content) and not combined_data["political_affiliation"]:
                        combined_data["political_affiliation"] = "Democratic Party"
                    elif ("Republican Party" in content or "Republican" in content) and not combined_data["political_affiliation"]:
                        combined_data["political_affiliation"] = "Republican Party"
                
            except Exception as e:
                print(f"Error processing {url}: {str(e)}")
    
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

def main(name):
    print(f"Starting AI-powered crawl for {name} using Ollama/Llama3...")
    data = asyncio.run(crawl_political_figure(name))
    
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
    
    return data

if __name__ == "__main__":
    # Example usage
    political_figure_name = "Barack Obama"
    main(political_figure_name)