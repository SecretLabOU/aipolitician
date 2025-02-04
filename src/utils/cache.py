"""Cache utilities for PoliticianAI."""

import logging
import pickle
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union

from sqlalchemy.orm import Session

from src.config import CACHE_EXPIRY_HOURS, RESPONSE_CACHE_SIZE
from src.database.models import Cache
from src.utils import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

class CacheManager:
    """Manager for handling response caching."""
    
    def __init__(self, db: Session):
        """
        Initialize cache manager.
        
        Args:
            db: Database session
        """
        self.db = db
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if found and not expired, None otherwise
        """
        try:
            # Get cache entry
            entry = (
                self.db.query(Cache)
                .filter(Cache.key == key)
                .first()
            )
            
            # Check if entry exists and is not expired
            if entry and entry.expires_at > datetime.utcnow():
                try:
                    return pickle.loads(entry.value.encode())
                except Exception as e:
                    self.logger.error(f"Error deserializing cache value: {str(e)}")
                    return None
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting from cache: {str(e)}")
            return None
    
    def set(
        self,
        key: str,
        value: Any,
        expiry_hours: Optional[int] = None
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            expiry_hours: Optional custom expiry time in hours
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Serialize value
            try:
                serialized = pickle.dumps(value).decode()
            except Exception as e:
                self.logger.error(f"Error serializing cache value: {str(e)}")
                return False
            
            # Calculate expiry time
            expiry = datetime.utcnow() + timedelta(
                hours=expiry_hours or CACHE_EXPIRY_HOURS
            )
            
            # Create or update cache entry
            entry = (
                self.db.query(Cache)
                .filter(Cache.key == key)
                .first()
            )
            
            if entry:
                entry.value = serialized
                entry.expires_at = expiry
                entry.updated_at = datetime.utcnow()
            else:
                entry = Cache(
                    key=key,
                    value=serialized,
                    expires_at=expiry
                )
                self.db.add(entry)
            
            # Enforce cache size limit
            self._enforce_cache_limit()
            
            self.db.commit()
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting cache: {str(e)}")
            self.db.rollback()
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete cache entry
            (
                self.db.query(Cache)
                .filter(Cache.key == key)
                .delete()
            )
            
            self.db.commit()
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting from cache: {str(e)}")
            self.db.rollback()
            return False
    
    def clear(self) -> bool:
        """
        Clear all cached values.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete all cache entries
            self.db.query(Cache).delete()
            self.db.commit()
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing cache: {str(e)}")
            self.db.rollback()
            return False
    
    def _enforce_cache_limit(self):
        """Enforce cache size limit by removing oldest entries."""
        try:
            # Get current cache size
            cache_size = self.db.query(Cache).count()
            
            if cache_size > RESPONSE_CACHE_SIZE:
                # Get oldest entries to remove
                to_remove = cache_size - RESPONSE_CACHE_SIZE
                oldest_entries = (
                    self.db.query(Cache)
                    .order_by(Cache.updated_at)
                    .limit(to_remove)
                )
                
                # Delete oldest entries
                for entry in oldest_entries:
                    self.db.delete(entry)
                
        except Exception as e:
            self.logger.error(f"Error enforcing cache limit: {str(e)}")

class Cache:
    """Simple in-memory cache with expiry."""
    
    def __init__(self, expiry_hours: Optional[int] = None):
        """
        Initialize cache.
        
        Args:
            expiry_hours: Optional custom expiry time in hours
        """
        self.cache: Dict[str, Dict[str, Union[Any, datetime]]] = {}
        self.expiry_hours = expiry_hours or CACHE_EXPIRY_HOURS
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if found and not expired, None otherwise
        """
        try:
            # Check if key exists and is not expired
            if key in self.cache:
                entry = self.cache[key]
                if entry["expires"] > datetime.utcnow():
                    return entry["value"]
                else:
                    # Remove expired entry
                    del self.cache[key]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting from cache: {str(e)}")
            return None
    
    def set(
        self,
        key: str,
        value: Any,
        expiry_hours: Optional[int] = None
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            expiry_hours: Optional custom expiry time in hours
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Calculate expiry time
            expiry = datetime.utcnow() + timedelta(
                hours=expiry_hours or self.expiry_hours
            )
            
            # Store value with expiry
            self.cache[key] = {
                "value": value,
                "expires": expiry
            }
            
            # Enforce cache size limit
            if len(self.cache) > RESPONSE_CACHE_SIZE:
                # Remove oldest entry
                oldest_key = min(
                    self.cache.keys(),
                    key=lambda k: self.cache[k]["expires"]
                )
                del self.cache[oldest_key]
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting cache: {str(e)}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if key in self.cache:
                del self.cache[key]
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting from cache: {str(e)}")
            return False
    
    def clear(self) -> bool:
        """
        Clear all cached values.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.cache.clear()
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing cache: {str(e)}")
            return False
