"""Cache management for the PoliticianAI project."""

import hashlib
import json
import pickle
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.config import CACHE_DATABASE_URL, CACHE_EXPIRY_HOURS
from src.utils.metrics import track_cache_access

class ResponseCache:
    """Cache for storing and retrieving responses."""
    
    def __init__(self, db_url: str = CACHE_DATABASE_URL):
        """
        Initialize cache with database connection.
        
        Args:
            db_url: SQLite database URL
        """
        self.engine = create_engine(db_url)
        self._create_cache_table()

    def _create_cache_table(self):
        """Create cache table if it doesn't exist."""
        with self.engine.connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS response_cache (
                    query_hash TEXT PRIMARY KEY,
                    response TEXT,
                    metadata TEXT,
                    timestamp DATETIME,
                    expiry DATETIME
                )
            """)

    def _compute_hash(self, data: Union[str, Dict]) -> str:
        """
        Compute hash for cache key.
        
        Args:
            data: Data to hash
            
        Returns:
            str: Hash value
        """
        if isinstance(data, dict):
            # Sort dictionary to ensure consistent hashing
            data = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()

    def get(self, query: Union[str, Dict]) -> Optional[Dict[str, Any]]:
        """
        Get cached response.
        
        Args:
            query: Query to look up
            
        Returns:
            Optional[Dict]: Cached response or None
        """
        query_hash = self._compute_hash(query)
        track_cache_access()  # Track cache request
        
        with Session(self.engine) as session:
            result = session.execute("""
                SELECT response, metadata, expiry
                FROM response_cache
                WHERE query_hash = :query_hash
            """, {"query_hash": query_hash}).fetchone()
            
            if result is None:
                track_cache_access(hit=False)
                return None
            
            response, metadata, expiry = result
            expiry = datetime.fromisoformat(expiry)
            
            # Check if cached response has expired
            if datetime.now() > expiry:
                self.delete(query)
                track_cache_access(hit=False)
                return None
            
            track_cache_access(hit=True)
            return {
                "response": json.loads(response),
                "metadata": json.loads(metadata)
            }

    def set(
        self,
        query: Union[str, Dict],
        response: Dict[str, Any],
        expiry_hours: Optional[int] = None
    ) -> bool:
        """
        Cache a response.
        
        Args:
            query: Query to cache
            response: Response to cache
            expiry_hours: Cache expiry time in hours
            
        Returns:
            bool: True if cached successfully
        """
        query_hash = self._compute_hash(query)
        expiry_hours = expiry_hours or CACHE_EXPIRY_HOURS
        expiry = datetime.now() + timedelta(hours=expiry_hours)
        
        try:
            with Session(self.engine) as session:
                session.execute("""
                    INSERT OR REPLACE INTO response_cache
                    (query_hash, response, metadata, timestamp, expiry)
                    VALUES (:query_hash, :response, :metadata, :timestamp, :expiry)
                """, {
                    "query_hash": query_hash,
                    "response": json.dumps(response.get("response", {})),
                    "metadata": json.dumps(response.get("metadata", {})),
                    "timestamp": datetime.now().isoformat(),
                    "expiry": expiry.isoformat()
                })
                session.commit()
            return True
        except Exception as e:
            print(f"Error caching response: {str(e)}")
            return False

    def delete(self, query: Union[str, Dict]) -> bool:
        """
        Delete cached response.
        
        Args:
            query: Query to delete
            
        Returns:
            bool: True if deleted successfully
        """
        query_hash = self._compute_hash(query)
        
        try:
            with Session(self.engine) as session:
                session.execute("""
                    DELETE FROM response_cache
                    WHERE query_hash = :query_hash
                """, {"query_hash": query_hash})
                session.commit()
            return True
        except Exception as e:
            print(f"Error deleting cached response: {str(e)}")
            return False

    def clear_expired(self) -> int:
        """
        Clear expired cache entries.
        
        Returns:
            int: Number of entries cleared
        """
        try:
            with Session(self.engine) as session:
                result = session.execute("""
                    DELETE FROM response_cache
                    WHERE expiry < :now
                """, {"now": datetime.now().isoformat()})
                session.commit()
                return result.rowcount
        except Exception as e:
            print(f"Error clearing expired cache: {str(e)}")
            return 0

    def clear_all(self) -> bool:
        """
        Clear all cache entries.
        
        Returns:
            bool: True if cleared successfully
        """
        try:
            with Session(self.engine) as session:
                session.execute("DELETE FROM response_cache")
                session.commit()
            return True
        except Exception as e:
            print(f"Error clearing cache: {str(e)}")
            return False

# Global cache instance
cache = ResponseCache()
