# Fact Checker Environment Setup

This document explains how to set up the necessary environment variables for the real-time fact-checking system.

## Required API Keys

The fact-checking system attempts to use several different APIs in a sequential, optimized manner to maximize free tier usage. You need to set up at least one of these to enable real fact-checking.

### Setting Up Your .env File

Create a `.env` file in the root directory of your project with the following API keys:

```
GOOGLE_FACT_CHECK_API_KEY=your_key_here
CLAIMBUSTER_API_KEY=your_key_here 
SERPER_API_KEY=your_key_here
NEWS_API_KEY=your_key_here
```

The system will automatically load these keys from your .env file using python-dotenv.

### API Usage Optimization

The system uses an intelligent API rotation algorithm that:

1. Tracks API usage to stay within free tier limits
2. Detects rate limiting and automatically switches to the next available API
3. Resets daily/monthly counters appropriately
4. Stores API usage state between runs to maintain continuity

It will try APIs in this order:
1. Google Fact Check API (most authoritative)
2. ClaimBuster API (specialized for claims)
3. Serper API (general search)
4. NewsAPI (news articles)
5. Basic web fallback (when all others are unavailable)

## How to Get API Keys

### Google Fact Check API

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the "Fact Check Tools API" for your project
4. Create an API key in the "Credentials" section
5. Add to your .env file:
   ```
   GOOGLE_FACT_CHECK_API_KEY=your_api_key_here
   ```

### ClaimBuster API

1. Go to [ClaimBuster](https://idir.uta.edu/claimbuster/) and register for an account
2. Request an API key through their contact form
3. Add to your .env file:
   ```
   CLAIMBUSTER_API_KEY=your_api_key_here
   ```

### Search APIs (Fallback Options)

The system will fall back to general search APIs if the fact-checking APIs are not available:

#### Serper API (Google Search API)

1. Go to [Serper.dev](https://serper.dev/) and sign up
2. Get your API key from the dashboard
3. Add to your .env file:
   ```
   SERPER_API_KEY=your_api_key_here
   ```

#### News API

1. Go to [NewsAPI.org](https://newsapi.org/) and sign up
2. Get your API key
3. Add to your .env file:
   ```
   NEWS_API_KEY=your_api_key_here
   ```

## Installation

1. Install the required packages:
   ```
   pip install -r requirements-fact-checker.txt
   ```

2. Create your `.env` file with at least one API key

3. Run your debate with fact-checking enabled:
   ```
   python -m scripts.test_debate --fact-check-enabled=True
   ```

## Testing

To test if your fact-checking setup is working correctly, you can run:

```python
from src.models.langgraph.debate.agents import check_claim_accuracy
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

result = check_claim_accuracy("Climate change is causing sea levels to rise")
print(result)
```

A successful setup will return real fact-checking results with actual sources.

## API Usage Monitoring

The system creates an `api_usage_state.json` file to track API usage. You can check this file to monitor your API usage patterns.

```json
{
  "google": {"daily_count": 5, "monthly_count": 25, "rate_limited_until": null},
  "claimbuster": {"daily_count": 0, "monthly_count": 0, "rate_limited_until": null},
  "serper": {"daily_count": 3, "monthly_count": 10, "rate_limited_until": null},
  "newsapi": {"daily_count": 0, "monthly_count": 0, "rate_limited_until": null}
}
``` 