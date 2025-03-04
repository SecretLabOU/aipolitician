"""
Database package for the Political RAG system.

This package provides functionality for storing and retrieving data
for the political debate system focused on Donald Trump and Joe Biden.
"""
from db.database import Database, get_database
from db.utils.embedding_utils import get_embedding_index

__all__ = [
    'Database',
    'get_database',
    'get_embedding_index',
]