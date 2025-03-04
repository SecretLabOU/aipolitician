"""
Utility functions for interacting with external APIs.

This module provides common functionality for retrieving data from
various APIs used across the Political RAG system.
"""
import json
import time
from typing import Any, Dict, List, Optional, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def create_session() -> requests.Session:
    """
    Create a requests session with retry capabilities.
    
    Returns:
        A requests session configured with retry logic.
    """
    session = requests.Session()
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session


def fetch_with_rate_limit(
    url: str, 
    params: Optional[Dict[str, Any]] = None, 
    headers: Optional[Dict[str, str]] = None,
    method: str = "GET",
    data: Optional[Dict[str, Any]] = None,
    rate_limit_delay: float = 1.0
) -> Dict[str, Any]:
    """
    Fetch data from an API with rate limiting.
    
    Args:
        url: The URL to fetch
        params: Query parameters
        headers: HTTP headers
        method: HTTP method (GET or POST)
        data: Request body for POST requests
        rate_limit_delay: Time to wait between requests (seconds)
        
    Returns:
        The JSON response as a dictionary
    """
    session = create_session()
    
    # Add default headers if not provided
    if headers is None:
        headers = {}
    
    if "User-Agent" not in headers:
        headers["User-Agent"] = "Political RAG Database Builder (https://github.com/your-username/aipolitician)"
    
    # Rate limiting delay
    time.sleep(rate_limit_delay)
    
    if method.upper() == "GET":
        response = session.get(url, params=params, headers=headers)
    elif method.upper() == "POST":
        response = session.post(url, params=params, headers=headers, json=data)
    else:
        raise ValueError(f"Unsupported HTTP method: {method}")
    
    response.raise_for_status()
    
    return response.json()


def fetch_wikipedia_data(title: str) -> Dict[str, Any]:
    """
    Fetch data about a Wikipedia page.
    
    Args:
        title: The title of the Wikipedia page
        
    Returns:
        The Wikipedia page data
    """
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts|pageimages|info|categories",
        "exintro": True,
        "explaintext": True,
        "inprop": "url",
        "pithumbsize": 500,
    }
    
    return fetch_with_rate_limit(url, params)


def fetch_congress_gov_data(congress: int, bill_number: str) -> Dict[str, Any]:
    """
    Fetch data about a bill from Congress.gov API.
    
    Args:
        congress: The congress number (e.g., 117)
        bill_number: The bill number (e.g., "hr1")
        
    Returns:
        Data about the bill
    """
    url = f"https://api.congress.gov/v3/bill/{congress}/{bill_number}"
    params = {
        "api_key": "DEMO_KEY",  # Replace with actual API key in production
        "format": "json",
    }
    
    return fetch_with_rate_limit(url, params)


def fetch_factcheck_data(claim: str) -> List[Dict[str, Any]]:
    """
    Fetch fact-check data for a claim.
    
    Note: This is a placeholder. There's no official API for FactCheck.org
    or PolitiFact. In production, you might use web scraping or a paid
    fact-checking API.
    
    Args:
        claim: The claim to fact-check
        
    Returns:
        A list of fact-checks for the claim
    """
    # This is a placeholder implementation
    # In a real implementation, you would use a fact-checking API or web scraping
    return []


def fetch_bea_economic_data(
    table_name: str, 
    frequency: str = "A", 
    start_year: int = 2016,
    end_year: int = 2023
) -> Dict[str, Any]:
    """
    Fetch economic data from the US Bureau of Economic Analysis.
    
    Args:
        table_name: The BEA table name (e.g., "T10101")
        frequency: Data frequency (A=Annual, Q=Quarterly, M=Monthly)
        start_year: Starting year
        end_year: Ending year
        
    Returns:
        Economic data
    """
    url = "https://apps.bea.gov/api/data"
    params = {
        "UserID": "DEMO_KEY",  # Replace with actual API key in production
        "method": "GetData",
        "DataSetName": "NIPA",
        "TableName": table_name,
        "Frequency": frequency,
        "Year": f"{start_year},{end_year}",
        "ResultFormat": "JSON",
    }
    
    return fetch_with_rate_limit(url, params)


def fetch_opensecrets_data(cid: str, cycle: int) -> Dict[str, Any]:
    """
    Fetch campaign finance data from OpenSecrets API.
    
    Args:
        cid: Candidate ID
        cycle: Election cycle (e.g., 2020)
        
    Returns:
        Campaign finance data
    """
    url = "https://www.opensecrets.org/api/"
    params = {
        "apikey": "DEMO_KEY",  # Replace with actual API key in production
        "method": "candSummary",
        "cid": cid,
        "cycle": cycle,
        "output": "json",
    }
    
    return fetch_with_rate_limit(url, params)