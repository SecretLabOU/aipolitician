# Browser-Based Fact Checker Setup

This document explains how to set up the browser-based fact-checking system using open source models.

## Overview

Instead of relying on external APIs that require API keys, this implementation uses browser automation via the `browser-use` library to directly search reputable fact-checking websites. Key features:

1. No API keys required for fact-checking services
2. Uses open source language models instead of OpenAI
3. Direct access to established fact-checking sources like Snopes, PolitiFact, FactCheck.org, etc.
4. Real-time results with current information

## Installation

1. Install the required packages:

```bash
pip install -r requirements-browser-fact-checker.txt
```

If you're using Ollama, make sure to install the langchain-ollama package:

```bash
pip install langchain-ollama
```

2. Install browser dependencies for Playwright (which is used by browser-use):

```bash
playwright install
```

3. Choose and set up your preferred open source model. You have three options:

### Option 1: Local models with Ollama (recommended)

1. Install Ollama from [ollama.ai](https://ollama.ai/)
2. Pull a model (llama3 recommended):
   ```bash
   ollama pull llama3
   ```
3. Configure environment variables:
   ```
   FACT_CHECK_MODEL_TYPE=ollama
   OLLAMA_MODEL=llama3
   ```

### Option 2: HuggingFace Inference API

1. Get a HuggingFace API token from [huggingface.co](https://huggingface.co/settings/tokens)
2. Configure environment variables:
   ```
   FACT_CHECK_MODEL_TYPE=huggingface_endpoint
   HUGGINGFACE_API_TOKEN=your_huggingface_token
   HUGGINGFACE_MODEL=mistralai/Mistral-7B-Instruct-v0.2
   ```

### Option 3: Local HuggingFace Models

1. Configure environment variables:
   ```
   FACT_CHECK_MODEL_TYPE=huggingface_local
   LOCAL_MODEL_PATH=TheBloke/Llama-2-7B-Chat-GGUF
   ```

## How It Works

The fact-checking process follows these steps:

1. When a claim needs to be verified, a browser instance is launched
2. The browser navigates to reputable fact-checking websites and searches for information related to the claim
3. It extracts conclusions, sources, and ratings from found fact-checks
4. Results are compiled into a structured format with:
   - Accuracy score (0-1)
   - Sources (with titles and URLs)
   - Corrections or context if available

If browser automation fails for any reason, the system falls back to a basic analysis that:
- Examines claim patterns (absolutes, statistics, etc.)
- Links to relevant authoritative sources based on topic
- Provides general fact-checking resources

## LLM Configuration

The system uses different LLM implementations depending on your selected model:

1. **Ollama**: Uses `OllamaLLM` from the `langchain_ollama` package
2. **HuggingFace**: Uses `HuggingFaceEndpoint` or `HuggingFacePipeline` from the `langchain_community` package

The implementation is in the `_browser_fact_check` function in `src/models/langgraph/debate/agents.py`.

## Browser Configuration

The system uses the `BrowserConfig` class from browser-use to configure the browser behavior. The current configuration sets the browser to run in headless mode for server environments. This can be customized by modifying the `_browser_fact_check` function in `src/models/langgraph/debate/agents.py`.

Example configuration:
```python
from browser_use import BrowserConfig

# Configure the browser to run headless
browser_context = BrowserConfig(headless=True)

# Pass to Agent as browser_context parameter
agent = Agent(
    task="Your task here",
    llm=your_llm,
    browser_context=browser_context
)

# Additional options are available, such as:
# browser_context = BrowserConfig(
#     headless=True,
#     locale="en-US",
#     highlight_elements=True
# )
```

## Testing

To test if your browser-based fact-checking setup is working correctly, you can run:

```python
from src.models.langgraph.debate.agents import check_claim_accuracy
import os

# Configure which model to use
os.environ["FACT_CHECK_MODEL_TYPE"] = "ollama"  # or "huggingface_endpoint" or "huggingface_local"
os.environ["OLLAMA_MODEL"] = "llama3"  # if using Ollama

result = check_claim_accuracy("Climate change is causing sea levels to rise")
print(result)
```

A successful setup will return real fact-checking results with actual sources.

## Troubleshooting

Common issues:

1. **Browser not starting**: Make sure you've run `playwright install`
2. **Model loading errors**: Ensure you've set the correct environment variables for your chosen model type
3. **Slow performance**: Using large language models and browser automation is resource-intensive - this is normal
4. **Poor quality results**: Try a more capable model - Llama 3, Mixtral, or GPT-4 open equivalents will produce the best results
5. **Memory issues**: If using local models, try a smaller or quantized model if you run out of memory
6. **Browser configuration errors**: If you encounter issues with the browser configuration, check the browser-use documentation for the latest API changes: https://docs.browser-use.com/
7. **Compatibility errors**: Make sure you're using the correct LangChain packages. For Ollama, use `langchain_ollama.OllamaLLM` instead of the deprecated `langchain.llms.Ollama` 