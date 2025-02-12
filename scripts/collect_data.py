#!/usr/bin/env python3
"""Script to initialize data for PoliticianAI."""

import logging
from pathlib import Path

from src.config import DATA_DIR
from src.database import Base, engine
from src.utils import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

def create_data_directories():
    """Create necessary data directories."""
    logger.info("Creating data directories...")
    
    # Create main data directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create models directory
    (DATA_DIR / 'models').mkdir(exist_ok=True)
    logger.info("Created models directory")

def initialize_database():
    """Initialize the database schema."""
    logger.info("Initializing database...")
    
    try:
        # Create all tables
        Base.metadata.create_all(engine)
        logger.info("Database tables created successfully")
        
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
