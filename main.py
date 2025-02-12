#!/usr/bin/env python3
"""Main entry point for PoliticianAI application."""

import logging
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api import router
from src.config import (
    API_HOST,
    API_PORT,
    API_WORKERS,
    DEBUG,
    LOGGING_CONFIG,
    get_database_url
)
from src.database import Base, init_db
from src.utils import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="PoliticianAI",
    description="AI system for political discourse simulation",
    version="1.0.0",
    debug=DEBUG
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router)

def main():
    """Run the application."""
    try:
        logger.info("Starting PoliticianAI application")
        
        # Initialize database
        logger.info("Initializing database...")
        engine = init_db()
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully")
        
        # Ensure required directories exist
        data_dir = Path("data")
        models_dir = Path("models")
        data_dir.mkdir(exist_ok=True)
        models_dir.mkdir(exist_ok=True)
        
        # Start the server
        uvicorn.run(
            "main:app",
            host=API_HOST,
            port=API_PORT,
            workers=API_WORKERS,
            reload=DEBUG,
            log_config=LOGGING_CONFIG
        )
        
    except Exception as e:
        logger.error(f"Application startup failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
