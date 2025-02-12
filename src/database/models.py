"""Database models for PoliticianAI."""

import datetime
from typing import Any, Dict, List

from sqlalchemy import (
    Column, DateTime, Float, ForeignKey, Integer, String, Text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

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
