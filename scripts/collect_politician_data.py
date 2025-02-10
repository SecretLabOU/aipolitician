#!/usr/bin/env python3
"""Script to collect and store politician data."""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import requests
from sqlalchemy.orm import Session

from src.config import DATA_DIR, POLITICAL_TOPICS
from src.database import Base, Session as DbSession, engine
from src.database.models import Politician, Statement, Topic, Vote
from src.utils import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

# API Configuration
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"

# Topic keywords mapping
TOPIC_KEYWORDS = {
    "Healthcare": ["health", "healthcare", "medical", "medicine", "hospital", "insurance", "medicare", "medicaid"],
    "Economy": ["economy", "economic", "jobs", "unemployment", "inflation", "market", "financial", "business"],
    "Education": ["education", "school", "student", "college", "university", "teacher", "learning"],
    "Immigration": ["immigration", "immigrant", "border", "visa", "asylum", "refugee"],
    "Climate Change": ["climate", "environment", "environmental", "green", "carbon", "emission"],
    "National Security": ["security", "defense", "military", "terrorism", "cyber", "intelligence"],
    "Gun Control": ["gun", "firearm", "weapon", "second amendment", "nra"],
    "Social Security": ["social security", "retirement", "pension", "elderly", "senior"],
    "Tax Policy": ["tax", "taxation", "irs", "revenue", "fiscal"],
    "Foreign Policy": ["foreign", "international", "diplomatic", "diplomacy", "global"],
    "Criminal Justice": ["crime", "criminal", "justice", "police", "law enforcement", "prison"],
    "Infrastructure": ["infrastructure", "roads", "bridges", "transportation", "construction"],
    "Energy Policy": ["energy", "oil", "gas", "renewable", "solar", "wind", "nuclear"],
    "Trade Policy": ["trade", "tariff", "export", "import", "commerce"],
    "Civil Rights": ["civil rights", "equality", "discrimination", "voting rights", "minority"]
}

class PoliticianDataCollector:
    """Collector for politician data."""
    
    def __init__(self):
        """Initialize data collector."""
        logger.info("Initializing PoliticianDataCollector")
        self.session = DbSession()
        self.setup_database()
    
    def setup_database(self):
        """Ensure database is initialized."""
        logger.info("Setting up database...")
        Base.metadata.create_all(engine)
        self._ensure_topics_exist()
    
    def _ensure_topics_exist(self):
        """Ensure all political topics exist in database."""
        logger.info("Ensuring topics exist...")
        for topic_name in POLITICAL_TOPICS:
            topic = self.session.query(Topic).filter_by(name=topic_name).first()
            if not topic:
                topic = Topic(name=topic_name)
                self.session.add(topic)
                logger.info(f"Added new topic: {topic_name}")
        self.session.commit()
        
        # Verify topics were created
        topics = self.session.query(Topic).all()
        logger.info(f"Total topics in database: {len(topics)}")
        for topic in topics:
            logger.info(f"Topic: {topic.name}")
    
    def collect_politician_basic_info(self) -> None:
        """Collect and store basic information for Trump and Biden."""
        logger.info("Collecting basic politician information...")
        politicians_info = {
            "Donald Trump": {
                "party": "Republican",
                "position": "Former President",
                "bio": "45th President of the United States (2017-2021). "
                      "Businessman and television personality before presidency."
            },
            "Joe Biden": {
                "party": "Democratic",
                "position": "President",
                "bio": "46th President of the United States (2021-present). "
                      "Former Vice President (2009-2017) and Senator from Delaware."
            }
        }
        
        for name, info in politicians_info.items():
            politician = self.session.query(Politician).filter_by(name=name).first()
            if not politician:
                politician = Politician(
                    name=name,
                    party=info["party"],
                    position=info["position"],
                    bio=info["bio"]
                )
                self.session.add(politician)
                logger.info(f"Added new politician: {name}")
            else:
                politician.party = info["party"]
                politician.position = info["position"]
                politician.bio = info["bio"]
                logger.info(f"Updated existing politician: {name}")
        
        self.session.commit()
        
        # Verify politicians were created
        politicians = self.session.query(Politician).all()
        logger.info(f"Total politicians in database: {len(politicians)}")
        for politician in politicians:
            logger.info(f"Politician: {politician.name} ({politician.party})")
    
    def collect_statements(self, politician_name: str) -> None:
        """
        Collect public statements for a politician.
        
        Args:
            politician_name: Name of the politician
        """
        logger.info(f"Collecting statements for {politician_name}...")
        
        if not NEWS_API_KEY:
            logger.error("NEWS_API_KEY not set")
            return
        else:
            logger.info("Using NEWS_API_KEY: " + NEWS_API_KEY[:5] + "...")
        
        politician = self.session.query(Politician).filter_by(name=politician_name).first()
        if not politician:
            logger.error(f"Politician not found: {politician_name}")
            return
        
        # Collect recent news statements
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": f'"{politician_name}"',
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": 100
        }
        headers = {"X-Api-Key": NEWS_API_KEY}
        
        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            articles = response.json().get("articles", [])
            logger.info(f"Found {len(articles)} articles for {politician_name}")
            
            statements_added = 0
            for article in articles:
                # Extract relevant quotes or content
                content = article.get("content", "")
                if not content:
                    continue
                
                # Determine topic using improved keyword matching
                topic = self._determine_topic(content)
                if not topic:
                    continue
                
                # Store statement
                statement = Statement(
                    politician_id=politician.id,
                    topic_id=topic.id,
                    content=content[:500],  # Limit content length
                    source=article.get("url"),
                    date=datetime.strptime(
                        article.get("publishedAt", ""), 
                        "%Y-%m-%dT%H:%M:%SZ"
                    )
                )
                self.session.add(statement)
                statements_added += 1
            
            self.session.commit()
            logger.info(f"Added {statements_added} statements for {politician_name}")
            
        except Exception as e:
            logger.error(f"Error collecting statements: {str(e)}")
            self.session.rollback()
    
    def _determine_topic(self, content: str) -> Optional[Topic]:
        """
        Determine the topic of a statement using keyword matching.
        
        Args:
            content: Statement content
            
        Returns:
            Matched Topic or None
        """
        content_lower = content.lower()
        
        # Try to match using keywords
        for topic_name, keywords in TOPIC_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in content_lower:
                    topic = self.session.query(Topic).filter_by(name=topic_name).first()
                    if topic:
                        return topic
        
        return None
    
    def collect_wikipedia_info(self, politician_name: str) -> None:
        """
        Collect information from Wikipedia for a politician.
        
        Args:
            politician_name: Name of the politician
        """
        logger.info(f"Collecting Wikipedia information for {politician_name}...")
        
        politician = self.session.query(Politician).filter_by(name=politician_name).first()
        if not politician:
            logger.error(f"Politician not found: {politician_name}")
            return
        
        try:
            # Get page content from Wikipedia API
            params = {
                "action": "query",
                "format": "json",
                "titles": politician_name,
                "prop": "extracts",
                "exintro": True,
                "explaintext": True
            }
            
            response = requests.get(WIKIPEDIA_API_URL, params=params)
            response.raise_for_status()
            
            # Extract page content
            pages = response.json()["query"]["pages"]
            page = next(iter(pages.values()))
            content = page.get("extract", "")
            
            # Parse content into statements
            sentences = content.split(". ")
            statements_added = 0
            
            for sentence in sentences:
                if not sentence.strip():
                    continue
                    
                # Determine topic using improved keyword matching
                topic = self._determine_topic(sentence)
                if not topic:
                    continue
                
                # Store statement
                statement = Statement(
                    politician_id=politician.id,
                    topic_id=topic.id,
                    content=sentence.strip(),
                    source="Wikipedia",
                    date=datetime.now(),
                    sentiment_score=0.0  # Neutral by default
                )
                self.session.add(statement)
                statements_added += 1
            
            self.session.commit()
            logger.info(f"Added {statements_added} Wikipedia statements for {politician_name}")
            
        except Exception as e:
            logger.error(f"Error collecting Wikipedia information: {str(e)}")
            self.session.rollback()
    
    def close(self):
        """Close database session."""
        self.session.close()

def main():
    """Main data collection function."""
    try:
        logger.info("Starting data collection...")
        collector = PoliticianDataCollector()
        
        # Collect basic information
        collector.collect_politician_basic_info()
        
        # Collect data for each politician
        politicians = ["Donald Trump", "Joe Biden"]
        for politician in politicians:
            collector.collect_statements(politician)
            collector.collect_wikipedia_info(politician)
        
        # Verify final data
        with collector.session as session:
            politicians = session.query(Politician).all()
            for politician in politicians:
                statement_count = session.query(Statement).filter_by(politician_id=politician.id).count()
                logger.info(f"{politician.name} has {statement_count} statements")
        
        logger.info("Data collection complete!")
        
    except Exception as e:
        logger.error(f"Data collection failed: {str(e)}")
        raise
    
    finally:
        collector.close()

if __name__ == "__main__":
    main()
