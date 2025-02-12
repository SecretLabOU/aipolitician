"""Utility functions for PoliticianAI."""

import logging

def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )

__all__ = ['setup_logging']
