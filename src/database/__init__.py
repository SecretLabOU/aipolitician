"""Database module for PoliticianAI."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.config import DB_CONFIG
from src.database.models import Base, ChatHistory

# Create session factory
Session = sessionmaker()

def init_db(database_url: str = None):
    """Initialize database with optional URL."""
    from src.config import get_database_url
    
    # Create database engine
    engine = create_engine(
        database_url or get_database_url(),
        **DB_CONFIG
    )
    
    # Bind session factory to engine
    Session.configure(bind=engine)
    
    return engine

__all__ = ['Base', 'ChatHistory', 'Session', 'init_db']
