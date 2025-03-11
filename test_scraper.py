import asyncio
import json
from crawl4ai import (
    AsyncWebCrawler, LLMExtractionStrategy, LLMConfig, 
    CrawlerRunConfig, RegexChunking
)
import requests
import time

async def test_extraction(name="Barack Obama"):
    # Just test with one URL to make it faster
    url = f"https://en.wikipedia.org/wiki/{name.replace(' ', '_')}"
    print(f"Testing extraction on: {url}")
    
    # Configure Ollama with explicit model parameters
    llm_config = LLMConfig(
        provider="ollama/llama3",
        base_url="http://localhost:11434"
    )
    
    # More explicit extraction prompt that forces JSON output
    extraction_strategy = LLMExtractionStrategy(
        llm_config=llm_config,
        prompt_template="""
        You are a data extraction assistant. Extract the following from the text:
        
        1. A short biography (1-2 sentences max)
        2. Birth date in MM/DD/YYYY format if available
        3. Political party/affiliation
        
        Format as valid JSON only. Use this exact format:
        
        ```json
        {
          "biography": "...",
          "birth_date": "...",
          "party": "..."
        }
        ```
        
        Focus only on {name}. Do not include explanations or notes.
        """,
        variables={"name": name},
        # Smaller chunks for faster processing
        chunk_token_threshold=1000,
        overlap_rate=0.1
    )
    
    # Simplified config
    crawler_config = CrawlerRunConfig(
        extraction_strategy=extraction_strategy,
        chunking_strategy=RegexChunking(patterns=[r"\n\n"]),
        word_count_threshold=25,
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
            
            # Debug the result
            print("\nResult attributes:", dir(result))
            
            # Check for extracted data
            if hasattr(result, 'extracted_data') and result.extracted_data:
                print("\n✅ SUCCESS! Extracted structured data:")
                print(json.dumps(result.extracted_data, indent=2))
                return True
            elif hasattr(result, 'markdown') and result.markdown:
                # Try to extract something from markdown
                print("\n⚠️ Got markdown but no structured data extraction")
                print(f"Markdown length: {len(result.markdown)} characters")
                # Show first 200 chars
                print(f"Preview: {result.markdown[:200]}...")
                
                # Try to find if there's any JSON in the response
                import re
                json_match = re.search(r'```json\s*(.*?)\s*```', result.markdown, re.DOTALL)
                if json_match:
                    print("\nFound JSON in markdown response:")
                    json_str = json_match.group(1)
                    print(json_str)
                    try:
                        json_data = json.loads(json_str)
                        print("Parsed JSON successfully")
                        return True
                    except:
                        print("Failed to parse JSON")
                
                # Add this to your test script to try direct extraction from markdown
                print("\nTrying direct extraction...")
                extracted_text = direct_extraction(result.markdown)
                print("Result:", extracted_text)
                
                return False
            else:
                print("\n❌ No data extracted")
                print("Result attributes:", dir(result))
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