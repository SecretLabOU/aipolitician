import asyncio
import json
from crawl4ai import (
    AsyncWebCrawler, LLMExtractionStrategy, LLMConfig, 
    CrawlerRunConfig, RegexChunking
)

async def test_extraction(name="Barack Obama"):
    """
    Small test function that only scrapes Wikipedia for quick testing.
    """
    # Just test with one URL to make it faster
    url = f"https://en.wikipedia.org/wiki/{name.replace(' ', '_')}"
    print(f"Testing extraction on: {url}")
    
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
                # Show first 200 chars
                print(f"Preview: {result.markdown[:200]}...")
                return False
            else:
                print("\n❌ No data extracted")
                return False
                
        except Exception as e:
            print(f"\n❌ Error during crawl: {str(e)}")
            import traceback
            traceback.print_exc()
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