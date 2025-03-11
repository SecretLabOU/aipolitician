import asyncio
import json
import requests
from crawl4ai import (
    AsyncWebCrawler, LLMExtractionStrategy, LLMConfig, 
    CrawlerRunConfig, RegexChunking
)
import time

# Direct extraction function using Ollama API
def direct_extraction(text, name="Barack Obama"):
    """Extract information directly using Ollama API"""
    # Take just first 4000 characters to keep it fast
    text = text[:4000] 
    
    prompt = f"""
    Extract the following about {name} from this text:
    
    1. Brief biography (2 sentences max)
    2. Birth date
    3. Political party
    
    Return valid JSON only in this format:
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
        print(f"Error with direct extraction: {e}")
        return None

async def test_extraction(name="Barack Obama"):
    """
    Small test function that only scrapes Wikipedia for quick testing.
    """
    # Just test with one URL to make it faster
    url = f"https://en.wikipedia.org/wiki/{name.replace(' ', '_')}"
    print(f"Testing extraction on: {url}")
    
    # Check Ollama connection first
    try:
        response = requests.get('http://localhost:11434/api/tags')
        print("Ollama is running and available models:", response.json())
    except:
        print("⚠️ Cannot connect to Ollama service")
    
    # Configure Ollama - same as main script
    llm_config = LLMConfig(
        provider="ollama/llama3",
        base_url="http://localhost:11434"
    )
    
    # Simplified extraction strategy - smaller prompt
    extraction_strategy = LLMExtractionStrategy(
        llm_config=llm_config,
        prompt_template="""
        Extract the following from the text about {name}:
        1. A one-sentence biography
        2. Birth date
        3. Political party
        
        Format as JSON: {{"biography": "...", "birth_date": "...", "party": "..."}}
        """,
        variables={"name": name},
        # Smaller chunks for faster processing
        chunk_token_threshold=2000,
        overlap_rate=0.1
    )
    
    # Simplified config
    crawler_config = CrawlerRunConfig(
        extraction_strategy=extraction_strategy,
        chunking_strategy=RegexChunking(patterns=[r"\n\n"]),
        word_count_threshold=30,  # Lower threshold
        cache_mode="bypass",
        verbose=True
    )
    
    # Execute single crawl
    async with AsyncWebCrawler() as crawler:
        try:
            print("Starting crawl...")
            start_time = asyncio.get_event_loop().time()
            
            result = await crawler.arun(url=url, config=crawler_config)
            
            end_time = asyncio.get_event_loop().time()
            duration = end_time - start_time
            print(f"Crawl completed in {duration:.2f} seconds")
            
            # Check for extracted data
            if hasattr(result, 'extracted_data') and result.extracted_data:
                print("\n✅ SUCCESS! Extracted structured data:")
                print(json.dumps(result.extracted_data, indent=2))
                return True
            elif hasattr(result, 'markdown') and result.markdown:
                print("\n⚠️ Got markdown but no structured data extraction")
                print(f"Markdown length: {len(result.markdown)} characters")
                
                # Try direct extraction as a fallback
                print("\nTrying direct extraction from first part of markdown...")
                extracted_text = direct_extraction(result.markdown, name)
                if extracted_text:
                    print("Result:", extracted_text)
                    
                    # Try to parse as JSON
                    try:
                        json_data = json.loads(extracted_text)
                        print("\nSuccessfully parsed as JSON:")
                        print(json.dumps(json_data, indent=2))
                        return True
                    except json.JSONDecodeError:
                        print("\nCouldn't parse direct extraction as JSON")
                        # Just extract what we got
                        print("Using text response instead")
                return False
            else:
                print("\n❌ No data extracted")
                return False
                
        except Exception as e:
            print(f"\n❌ Error during crawl: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def check_ollama_gpu():
    try:
        # Simple query to check model info
        response = requests.post('http://localhost:11434/api/show', 
                                json={'name': 'llama3'})
        model_info = response.json()
        print("Model info:", model_info)
        
        # Run a quick benchmark to confirm GPU usage
        print("Running quick inference to test GPU...")
        start = time.time()
        response = requests.post('http://localhost:11434/api/generate',
                               json={'model': 'llama3', 
                                     'prompt': 'Write a brief sentence about AI.'})
        duration = time.time() - start
        print(f"Inference completed in {duration:.2f} seconds")
        
        # Quick inference should be under 1-2 seconds on GPU, much longer on CPU
        if duration < 3.0:
            print("✅ Inference speed suggests GPU is being used")
        else:
            print("⚠️ Inference seems slow, might be using CPU only")
            
        return True
    except Exception as e:
        print(f"Error checking Ollama: {e}")
        return False

def main():
    print("=== STARTING MINIMAL SCRAPER TEST ===")
    print("Testing Ollama/Llama3 connection and extraction...")
    success = asyncio.run(test_extraction())
    
    print("\n=== TEST SUMMARY ===")
    if success:
        print("✅ Test passed - extraction working correctly")
    else:
        print("⚠️ Test completed with issues - check logs above")
    
    print("\nIf this test works, your full scraper should work too!")

if __name__ == "__main__":
    main() 