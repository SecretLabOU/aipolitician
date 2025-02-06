"""Database models for PoliticianAI."""

import datetime
from typing import Any, Dict, List

from sqlalchemy import (
    Column, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.sql import func

from src.config import DATABASE_URL

# Create database engine with appropriate settings for PostgreSQL
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

# Create base class for declarative models
Base = declarative_base()

class Topic(Base):
    """Political topic model."""
    
    __tablename__ = "topics"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())
    
    # Relationships
    statements = relationship("Statement", back_populates="topic")

class Politician(Base):
    """Politician model."""
    
    __tablename__ = "politicians"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    party = Column(String(50))
    position = Column(String(100))
    bio = Column(Text)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())
    
    # Relationships
    statements = relationship("Statement", back_populates="politician")
    votes = relationship("Vote", back_populates="politician")

class Statement(Base):
    """Political statement model."""
    
    __tablename__ = "statements"
    
    id = Column(Integer, primary_key=True)
    politician_id = Column(Integer, ForeignKey("politicians.id"), nullable=False)
    topic_id = Column(Integer, ForeignKey("topics.id"), nullable=False)
    content = Column(Text, nullable=False)
    source = Column(String(200))
    date = Column(DateTime)
    sentiment_score = Column(Float)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())
    
    # Relationships
    politician = relationship("Politician", back_populates="statements")
    topic = relationship("Topic", back_populates="statements")

class Vote(Base):
    """Voting record model."""
    
    __tablename__ = "votes"
    
    id = Column(Integer, primary_key=True)
    politician_id = Column(Integer, ForeignKey("politicians.id"), nullable=False)
    bill_number = Column(String(50), nullable=False)
    bill_title = Column(String(200))
    vote = Column(String(20), nullable=False)  # 'yea', 'nay', 'present', 'absent'
    date = Column(DateTime, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())
    
    # Relationships
    politician = relationship("Politician", back_populates="votes")

class ChatHistory(Base):
    """Chat history model."""
    
    __tablename__ = "chat_history"
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(100), nullable=False)
    user_input = Column(Text, nullable=False)
    system_response = Column(Text, nullable=False)
    sentiment_score = Column(Float)
    context_topics = Column(String(200))  # Comma-separated topic IDs
    created_at = Column(DateTime, server_default=func.now())

class Cache(Base):
    """Cache model for storing response cache."""
    
    __tablename__ = "cache"
    
    id = Column(Integer, primary_key=True)
    key = Column(String(200), unique=True, nullable=False)
    value = Column(Text, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())
