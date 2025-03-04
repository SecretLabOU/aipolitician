"""
Utility functions for web scraping.

This module provides common functionality for scraping websites
without APIs to collect data for the Political RAG system.
"""
import time
from typing import Dict, List, Optional, Union, Any

import requests
from bs4 import BeautifulSoup


def get_soup(url: str, delay: float = 1.0) -> BeautifulSoup:
    """
    Get a BeautifulSoup object for a URL.
    
    Args:
        url: The URL to scrape
        delay: Time to wait before making the request (seconds)
        
    Returns:
        A BeautifulSoup object
    """
    # Add a delay to be respectful of the website
    time.sleep(delay)
    
    headers = {
        "User-Agent": "Political RAG Database Builder (https://github.com/your-username/aipolitician)"
    }
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    return BeautifulSoup(response.text, "html.parser")


def scrape_politifact(person: str, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Scrape fact-checks for a person from PolitiFact.
    
    Args:
        person: The person to search for (e.g., "Donald Trump")
        limit: Maximum number of fact-checks to retrieve
        
    Returns:
        A list of fact-checks
    """
    # Convert spaces to hyphens and make lowercase
    person_slug = person.lower().replace(" ", "-")
    url = f"https://www.politifact.com/personalities/{person_slug}/"
    
    soup = get_soup(url)
    
    fact_checks = []
    statement_items = soup.select(".m-statement__content")[:limit]
    
    for item in statement_items:
        # Extract statement text
        statement_text_elem = item.select_one(".m-statement__quote")
        statement = statement_text_elem.text.strip() if statement_text_elem else ""
        
        # Extract ruling
        ruling_elem = item.select_one(".m-statement__meter")
        ruling = ruling_elem.get("data-value") if ruling_elem else "Unknown"
        
        # Extract date
        date_elem = item.select_one(".m-statement__date")
        date = date_elem.text.strip() if date_elem else ""
        
        # Extract source link
        link_elem = item.select_one("a.m-statement__link")
        link = link_elem["href"] if link_elem else ""
        
        fact_checks.append({
            "statement": statement,
            "ruling": ruling,
            "date": date,
            "source": f"https://www.politifact.com{link}" if link.startswith("/") else link,
            "person": person
        })
    
    return fact_checks


def scrape_american_presidency_project(
    president: str,
    doc_type: str = "speeches",
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Scrape documents from The American Presidency Project.
    
    Args:
        president: The president's name (e.g., "Donald Trump")
        doc_type: The type of document (speeches, statements, executive-orders)
        limit: Maximum number of documents to retrieve
        
    Returns:
        A list of documents
    """
    # This is a simplified implementation
    # The actual implementation would need to handle pagination and different document types
    
    # Convert president name to format used in URL (e.g., "donald-j-trump")
    president_slug = president.lower().replace(" ", "-")
    if "trump" in president_slug:
        president_slug = "donald-j-trump"
    elif "biden" in president_slug:
        president_slug = "joseph-r-biden"
    
    url = f"https://www.presidency.ucsb.edu/advanced-search?field-keywords=&field-keywords2=&field-keywords3=&from%5Bdate%5D=&to%5Bdate%5D=&person2={president_slug}&category2%5B%5D={doc_type}"
    
    soup = get_soup(url)
    
    documents = []
    items = soup.select(".views-row")[:limit]
    
    for item in items:
        # Extract title
        title_elem = item.select_one("h2 a")
        title = title_elem.text.strip() if title_elem else ""
        
        # Extract date
        date_elem = item.select_one(".date-display-single")
        date = date_elem.text.strip() if date_elem else ""
        
        # Extract link
        link = title_elem["href"] if title_elem else ""
        
        documents.append({
            "title": title,
            "date": date,
            "source": f"https://www.presidency.ucsb.edu{link}" if link.startswith("/") else link,
            "president": president,
            "doc_type": doc_type
        })
    
    return documents


def scrape_congressional_record(
    person: str,
    chamber: str = "senate",
    limit: int = 50
) -> List[Dict[str, Any]]:
    """
    Scrape the Congressional Record for a person's statements.
    
    Args:
        person: The person's name (e.g., "Joe Biden")
        chamber: The chamber (senate or house)
        limit: Maximum number of records to retrieve
        
    Returns:
        A list of congressional records
    """
    # This is a placeholder implementation
    # The actual implementation would scrape from congress.gov
    
    # For demonstration purposes only
    return []


def scrape_pew_research(
    topic: str = "presidential-approval",
    limit: int = 20
) -> List[Dict[str, Any]]:
    """
    Scrape polling data from Pew Research Center.
    
    Args:
        topic: The topic to search for
        limit: Maximum number of polls to retrieve
        
    Returns:
        A list of polls
    """
    # This is a placeholder implementation
    # The actual implementation would need to navigate Pew Research's site structure
    
    url = f"https://www.pewresearch.org/topic/{topic}/"
    
    soup = get_soup(url)
    
    polls = []
    items = soup.select("article.article-card")[:limit]
    
    for item in items:
        # Extract title
        title_elem = item.select_one("h3.article-card__title a")
        title = title_elem.text.strip() if title_elem else ""
        
        # Extract date
        date_elem = item.select_one(".article-card__date")
        date = date_elem.text.strip() if date_elem else ""
        
        # Extract link
        link = title_elem["href"] if title_elem else ""
        
        polls.append({
            "title": title,
            "date": date,
            "source": link,
            "topic": topic
        })
    
    return polls