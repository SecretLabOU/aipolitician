"""Database module for PoliticianAI."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.config import DATABASE_URL
from src.database.models import Base, ChatHistory

# Create database engine
engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,
    echo=False
)

# Create session factory
Session = sessionmaker(bind=engine)

__all__ = ['Base', 'ChatHistory', 'Session']
