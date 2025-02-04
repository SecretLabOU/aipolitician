from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

Base = declarative_base()

class Politician(Base):
    __tablename__ = "politicians"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    party = Column(String)
    current_position = Column(String)
    bio_summary = Column(Text)
    
    # Relationships
    policy_positions = relationship("PolicyPosition", back_populates="politician")
    voting_records = relationship("VotingRecord", back_populates="politician")
    statements = relationship("Statement", back_populates="politician")

class PolicyPosition(Base):
    __tablename__ = "policy_positions"

    id = Column(Integer, primary_key=True)
    politician_id = Column(Integer, ForeignKey("politicians.id"))
    topic = Column(String, nullable=False)
    position = Column(Text, nullable=False)
    source = Column(String)
    date_updated = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    politician = relationship("Politician", back_populates="policy_positions")

class VotingRecord(Base):
    __tablename__ = "voting_records"

    id = Column(Integer, primary_key=True)
    politician_id = Column(Integer, ForeignKey("politicians.id"))
    bill_name = Column(String, nullable=False)
    vote = Column(String, nullable=False)
    date = Column(DateTime)
    topic = Column(String)
    source = Column(String)
    
    # Relationship
    politician = relationship("Politician", back_populates="voting_records")

class Statement(Base):
    __tablename__ = "statements"

    id = Column(Integer, primary_key=True)
    politician_id = Column(Integer, ForeignKey("politicians.id"))
    content = Column(Text, nullable=False)
    date = Column(DateTime)
    source_type = Column(String)  # speech, tweet, interview, etc.
    embedding_file = Column(String)  # Path to embedding file
    sentiment_score = Column(Float)
    
    # Relationship
    politician = relationship("Politician", back_populates="statements")

class ResponseCache(Base):
    __tablename__ = "response_cache"

    id = Column(Integer, primary_key=True)
    query_hash = Column(String, unique=True)
    response = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    expiry = Column(DateTime)

def init_db(db_path: str = "data/main.db"):
    """Initialize the database and create tables"""
    engine = create_engine(f"sqlite:///{db_path}", echo=True)
    Base.metadata.create_all(engine)
    
    # Create session factory
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return engine, SessionLocal

# Database dependency
def get_db(session_factory):
    db = session_factory()
    try:
        yield db
    finally:
        db.close()
