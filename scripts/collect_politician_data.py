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
PROPUBLICA_API_KEY = os.getenv("PROPUBLICA_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

class PoliticianDataCollector:
    """Collector for politician data."""
    
    def __init__(self):
        """Initialize data collector."""
        self.session = DbSession()
        self.setup_database()
    
    def setup_database(self):
        """Ensure database is initialized."""
        Base.metadata.create_all(engine)
        self._ensure_topics_exist()
    
    def _ensure_topics_exist(self):
        """Ensure all political topics exist in database."""
        for topic_name in POLITICAL_TOPICS:
            topic = self.session.query(Topic).filter_by(name=topic_name).first()
            if not topic:
                topic = Topic(name=topic_name)
                self.session.add(topic)
        self.session.commit()
    
    def collect_politician_basic_info(self) -> None:
        """Collect and store basic information for Trump and Biden."""
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
            else:
                politician.party = info["party"]
                politician.position = info["position"]
                politician.bio = info["bio"]
        
        self.session.commit()
        logger.info("Basic politician information stored")
    
    def collect_statements(self, politician_name: str) -> None:
        """
        Collect public statements for a politician.
        
        Args:
            politician_name: Name of the politician
        """
        if not NEWS_API_KEY:
            logger.error("NEWS_API_KEY not set")
            return
        
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
            
            for article in articles:
                # Extract relevant quotes or content
                content = article.get("content", "")
                if not content:
                    continue
                
                # Determine topic (simple keyword matching for now)
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
            
            self.session.commit()
            logger.info(f"Collected statements for {politician_name}")
            
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
        for topic in self.session.query(Topic).all():
            if topic.name.lower() in content_lower:
                return topic
        return None
    
    def collect_voting_records(self, politician_name: str) -> None:
        """
        Collect voting records for a politician.
        
        Args:
            politician_name: Name of the politician
        """
        if not PROPUBLICA_API_KEY:
            logger.error("PROPUBLICA_API_KEY not set")
            return
        
        politician = self.session.query(Politician).filter_by(name=politician_name).first()
        if not politician:
            logger.error(f"Politician not found: {politician_name}")
            return
        
        # ProPublica Congress API endpoints
        # Note: This is simplified - would need proper member ID lookup
        url = "https://api.propublica.org/congress/v1/members/{member_id}/votes.json"
        headers = {"X-API-Key": PROPUBLICA_API_KEY}
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            votes_data = response.json().get("results", [])
            
            for vote_data in votes_data:
                vote = Vote(
                    politician_id=politician.id,
                    bill_number=vote_data.get("bill", {}).get("number", ""),
                    bill_title=vote_data.get("description", ""),
                    vote=vote_data.get("position", ""),
                    date=datetime.strptime(
                        vote_data.get("date", ""), 
                        "%Y-%m-%d"
                    )
                )
                self.session.add(vote)
            
            self.session.commit()
            logger.info(f"Collected voting records for {politician_name}")
            
        except Exception as e:
            logger.error(f"Error collecting voting records: {str(e)}")
            self.session.rollback()
    
    def close(self):
        """Close database session."""
        self.session.close()

def main():
    """Main data collection function."""
    try:
        collector = PoliticianDataCollector()
        
        # Collect basic information
        collector.collect_politician_basic_info()
        
        # Collect data for each politician
        politicians = ["Donald Trump", "Joe Biden"]
        for politician in politicians:
            collector.collect_statements(politician)
            collector.collect_voting_records(politician)
        
        logger.info("Data collection complete!")
        
    except Exception as e:
        logger.error(f"Data collection failed: {str(e)}")
        raise
    
    finally:
        collector.close()

if __name__ == "__main__":
    main()
