"""
Base database module for the Political RAG system.

This module provides a common interface for all databases in the system.
It handles SQLite database connections and basic operations.
"""
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Any, Generator, Optional, Tuple, Union

from db.config import DATABASE_PATHS


class Database:
    """Base class for all databases in the system."""
    
    def __init__(self, db_name: str):
        """
        Initialize the database.
        
        Args:
            db_name: The name of the database (corresponds to keys in DATABASE_PATHS)
        """
        self.db_path = DATABASE_PATHS.get(db_name)
        if not self.db_path:
            raise ValueError(f"Unknown database name: {db_name}")
        
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize the database if it doesn't exist
        self.initialize_db()
    
    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Get a connection to the database.
        
        Returns:
            A connection to the database.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Return rows as dict-like objects
        try:
            yield conn
        finally:
            conn.close()
    
    def initialize_db(self) -> None:
        """
        Initialize the database with the required schema.
        This method should be overridden by subclasses.
        """
        pass
    
    def execute_query(self, query: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
        """
        Execute a query and return the results.
        
        Args:
            query: The SQL query to execute
            params: Parameters for the query
            
        Returns:
            A list of rows as dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            results = cursor.fetchall()
            return [dict(row) for row in results]
    
    def execute_update(self, query: str, params: Optional[Tuple] = None) -> int:
        """
        Execute an update query and return the number of affected rows.
        
        Args:
            query: The SQL query to execute
            params: Parameters for the query
            
        Returns:
            The number of affected rows
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            conn.commit()
            return cursor.rowcount
    
    def execute_many(self, query: str, params_list: List[Tuple]) -> int:
        """
        Execute the same query with different parameters for each and commit.
        
        Args:
            query: The SQL query to execute
            params_list: List of parameter tuples
            
        Returns:
            The number of affected rows
        """
        if not params_list:
            return 0
            
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            conn.commit()
            return cursor.rowcount
    
    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.
        
        Args:
            table_name: The name of the table to check
            
        Returns:
            True if the table exists, False otherwise
        """
        query = """
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name=?
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (table_name,))
            return cursor.fetchone() is not None


# Factory function to get a database instance
def get_database(db_name: str) -> Database:
    """
    Get a database instance by name.
    
    Args:
        db_name: The name of the database to get
        
    Returns:
        A Database instance
    """
    # Import specific database classes here to avoid circular imports
    from db.schemas.biography_db import BiographyDatabase
    from db.schemas.policy_db import PolicyDatabase
    from db.schemas.voting_record_db import VotingRecordDatabase
    from db.schemas.speech_db import SpeechDatabase
    
    # Map of database names to classes that currently exist
    db_classes = {
        'biography': BiographyDatabase,
        'policy': PolicyDatabase,
        'voting_record': VotingRecordDatabase,
        'public_statements': SpeechDatabase,  # Using SpeechDatabase for public_statements
    }
    
    # Note: The following databases are planned but not implemented yet:
    # - fact_check
    # - timeline
    # - legislative
    # - campaign_promises
    # - executive_actions
    # - media_coverage
    # - public_opinion
    # - controversies
    # - policy_comparison
    # - judicial_appointments
    # - foreign_policy
    # - economic_metrics
    # - charity
    
    db_class = db_classes.get(db_name)
    if not db_class:
        raise ValueError(f"Unknown database name: {db_name}")
    
    return db_class(db_name)