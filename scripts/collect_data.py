#!/usr/bin/env python3
"""Script to collect and initialize data for PoliticianAI."""

import logging
import os
from pathlib import Path

from src.config import DATA_DIR, LOGGING_CONFIG, POLITICAL_TOPICS
from src.database import Base, Session, engine
from src.utils import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

def create_data_directories():
    """Create necessary data directories."""
    logger.info("Creating data directories...")
    
    # Create main data directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for different data types
    subdirs = [
        'raw',          # Raw data files
        'processed',    # Processed data files
        'embeddings',   # Vector embeddings
        'cache',        # Cache files
        'exports'       # Data exports
    ]
    
    for subdir in subdirs:
        (DATA_DIR / subdir).mkdir(exist_ok=True)
        logger.info(f"Created directory: {subdir}")

def initialize_database():
    """Initialize the database schema."""
    logger.info("Initializing database...")
    
    try:
        # Create all tables
        Base.metadata.create_all(engine)
        logger.info("Database tables created successfully")
        
        # Initialize session
        session = Session()
        
        try:
            # Add any initial data here
            # For example, adding political topics
            # topic_table.insert().values([{'name': topic} for topic in POLITICAL_TOPICS])
            
            session.commit()
            logger.info("Initial data added successfully")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding initial data: {str(e)}")
            raise
        
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise

def main():
    """Main data initialization function."""
    try:
        logger.info("Starting data initialization...")
        
        # Create data directories
        create_data_directories()
        
        # Initialize database
        initialize_database()
        
        logger.info("Data initialization complete!")
        
    except Exception as e:
        logger.error(f"Data initialization failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
