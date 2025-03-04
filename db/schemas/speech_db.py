"""
Speech Database Schema.

This module defines the schema for the Speech Database, which stores
transcripts and metadata for speeches, debates, and interviews.
"""
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from db.database import Database
from db.utils.embedding_utils import get_embedding_index


class SpeechDatabase(Database):
    """Speech Database for storing speech transcripts and metadata."""
    
    def __init__(self, db_name: str = "speech"):
        """
        Initialize the Speech Database.
        
        Args:
            db_name: The name of the database
        """
        super().__init__(db_name)
        self.embedding_index = get_embedding_index("speech")
    
    def initialize_db(self) -> None:
        """Initialize the database with the required schema."""
        with self.get_connection() as conn:
            # Create speeches table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS speeches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                politician_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                event_type TEXT NOT NULL,
                speech_date TEXT,
                location TEXT,
                summary TEXT,
                source_url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (politician_id) REFERENCES politicians (id)
            )
            """)
            
            # Create speech_segments table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS speech_segments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                speech_id INTEGER NOT NULL,
                segment_order INTEGER NOT NULL,
                segment_text TEXT NOT NULL,
                speaker TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (speech_id) REFERENCES speeches (id)
            )
            """)
            
            # Create speech_topics table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS speech_topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                speech_id INTEGER NOT NULL,
                topic TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (speech_id) REFERENCES speeches (id)
            )
            """)
            
            conn.commit()
    
    def add_speech(
        self,
        politician_id: int,
        title: str,
        event_type: str,
        speech_date: Optional[str] = None,
        location: Optional[str] = None,
        summary: Optional[str] = None,
        source_url: Optional[str] = None,
    ) -> int:
        """
        Add a speech to the database.
        
        Args:
            politician_id: The ID of the politician
            title: The title of the speech
            event_type: The type of event (e.g., "debate", "interview", "rally")
            speech_date: The date of the speech (YYYY-MM-DD)
            location: The location where the speech was given
            summary: A summary of the speech
            source_url: URL to the source of the information
            
        Returns:
            The ID of the newly created speech
        """
        query = """
        INSERT INTO speeches (
            politician_id, title, event_type, speech_date, location, summary, source_url
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (
                politician_id, title, event_type, speech_date, location, summary, source_url
            ))
            conn.commit()
            
            speech_id = cursor.lastrowid
            
            # Add to embedding index if there's a summary
            if summary:
                self.embedding_index.add(
                    [summary],
                    [{
                        "speech_id": speech_id,
                        "title": title,
                        "event_type": event_type
                    }]
                )
                self.embedding_index.save()
            
            return speech_id
    
    def add_speech_segment(
        self,
        speech_id: int,
        segment_order: int,
        segment_text: str,
        speaker: Optional[str] = None,
    ) -> int:
        """
        Add a segment to a speech.
        
        Args:
            speech_id: The ID of the speech
            segment_order: The order of the segment in the speech
            segment_text: The text of the segment
            speaker: The name of the speaker (if a debate or interview)
            
        Returns:
            The ID of the newly created segment
        """
        query = """
        INSERT INTO speech_segments (
            speech_id, segment_order, segment_text, speaker
        ) VALUES (?, ?, ?, ?)
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (
                speech_id, segment_order, segment_text, speaker
            ))
            conn.commit()
            
            # Add to embedding index
            speech = self.get_speech_by_id(speech_id)
            if speech:
                self.embedding_index.add(
                    [segment_text],
                    [{
                        "segment_id": cursor.lastrowid,
                        "speech_id": speech_id,
                        "title": speech["title"],
                        "speaker": speaker
                    }]
                )
                self.embedding_index.save()
            
            return cursor.lastrowid
    
    def add_speech_topic(
        self,
        speech_id: int,
        topic: str,
    ) -> int:
        """
        Add a topic for a speech.
        
        Args:
            speech_id: The ID of the speech
            topic: The topic
            
        Returns:
            The ID of the newly created speech topic
        """
        query = """
        INSERT INTO speech_topics (
            speech_id, topic
        ) VALUES (?, ?)
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (speech_id, topic))
            conn.commit()
            return cursor.lastrowid
    
    def get_speech_by_id(self, speech_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a speech by ID.
        
        Args:
            speech_id: The speech's ID
            
        Returns:
            The speech's data or None if not found
        """
        query = "SELECT * FROM speeches WHERE id = ?"
        results = self.execute_query(query, (speech_id,))
        return results[0] if results else None
    
    def get_speeches_by_politician(self, politician_id: int) -> List[Dict[str, Any]]:
        """
        Get speeches for a politician.
        
        Args:
            politician_id: The politician's ID
            
        Returns:
            A list of speeches
        """
        query = "SELECT * FROM speeches WHERE politician_id = ? ORDER BY speech_date DESC"
        return self.execute_query(query, (politician_id,))
    
    def get_speeches_by_event_type(
        self, 
        event_type: str,
        politician_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get speeches of a specific event type.
        
        Args:
            event_type: The type of event
            politician_id: Optional filter by politician
            
        Returns:
            A list of speeches
        """
        if politician_id:
            query = """
            SELECT * FROM speeches 
            WHERE event_type = ? AND politician_id = ? 
            ORDER BY speech_date DESC
            """
            return self.execute_query(query, (event_type, politician_id))
        else:
            query = "SELECT * FROM speeches WHERE event_type = ? ORDER BY speech_date DESC"
            return self.execute_query(query, (event_type,))
    
    def get_speech_segments(self, speech_id: int) -> List[Dict[str, Any]]:
        """
        Get segments for a speech.
        
        Args:
            speech_id: The speech's ID
            
        Returns:
            A list of speech segments
        """
        query = "SELECT * FROM speech_segments WHERE speech_id = ? ORDER BY segment_order"
        return self.execute_query(query, (speech_id,))
    
    def get_speech_topics(self, speech_id: int) -> List[str]:
        """
        Get topics for a speech.
        
        Args:
            speech_id: The speech's ID
            
        Returns:
            A list of topics
        """
        query = "SELECT topic FROM speech_topics WHERE speech_id = ?"
        results = self.execute_query(query, (speech_id,))
        return [result["topic"] for result in results]
    
    def get_speeches_by_topic(self, topic: str) -> List[Dict[str, Any]]:
        """
        Get speeches for a specific topic.
        
        Args:
            topic: The topic
            
        Returns:
            A list of speeches
        """
        query = """
        SELECT s.*
        FROM speeches s
        JOIN speech_topics st ON s.id = st.speech_id
        WHERE st.topic = ?
        ORDER BY s.speech_date DESC
        """
        return self.execute_query(query, (topic,))
    
    def get_full_speech_text(self, speech_id: int) -> str:
        """
        Get the full text of a speech.
        
        Args:
            speech_id: The speech's ID
            
        Returns:
            The full text of the speech
        """
        segments = self.get_speech_segments(speech_id)
        
        if not segments:
            return ""
        
        # If there are speakers, format differently
        if any(segment["speaker"] for segment in segments):
            speech_text = ""
            for segment in segments:
                speaker = segment["speaker"] or "Unknown"
                speech_text += f"{speaker}: {segment['segment_text']}\n\n"
            return speech_text.strip()
        else:
            # Just concatenate the segments
            return "\n\n".join(segment["segment_text"] for segment in segments)
    
    def search_speeches(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the speech database for content related to the query.
        
        Args:
            query: The search query
            top_k: Number of results to return
            
        Returns:
            A list of search results
        """
        return self.embedding_index.search(query, top_k=top_k)
