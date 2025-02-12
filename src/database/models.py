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

class ChatHistory(Base):
    """Chat history model."""
    
    __tablename__ = "chat_history"
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(100), nullable=False)
    user_input = Column(Text, nullable=False)
    system_response = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
