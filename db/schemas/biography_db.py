"""
Biography Database Schema.

This module defines the schema for the Biography Database, which stores
personal and professional biographical details about politicians.
"""
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from db.database import Database
from db.utils.api_utils import fetch_wikipedia_data
from db.utils.embedding_utils import get_embedding_index


class BiographyDatabase(Database):
    """Biography Database for storing biographical information."""
    
    def __init__(self, db_name: str = "biography"):
        """
        Initialize the Biography Database.
        
        Args:
            db_name: The name of the database
        """
        super().__init__(db_name)
        self.embedding_index = get_embedding_index("biography")
    
    def initialize_db(self) -> None:
        """Initialize the database with the required schema."""
        with self.get_connection() as conn:
            # Create politicians table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS politicians (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                full_name TEXT,
                birth_date TEXT,
                birth_place TEXT,
                party TEXT,
                current_office TEXT,
                image_url TEXT,
                wikipedia_url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # Create biographical_events table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS biographical_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                politician_id INTEGER NOT NULL,
                event_date TEXT,
                event_type TEXT NOT NULL,
                event_description TEXT NOT NULL,
                source_url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (politician_id) REFERENCES politicians (id)
            )
            """)
            
            # Create education table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS education (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                politician_id INTEGER NOT NULL,
                institution TEXT NOT NULL,
                degree TEXT,
                field_of_study TEXT,
                start_year INTEGER,
                end_year INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (politician_id) REFERENCES politicians (id)
            )
            """)
            
            # Create career table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS career (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                politician_id INTEGER NOT NULL,
                organization TEXT NOT NULL,
                role TEXT NOT NULL,
                start_year INTEGER,
                end_year INTEGER,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (politician_id) REFERENCES politicians (id)
            )
            """)
            
            # Create family table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS family (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                politician_id INTEGER NOT NULL,
                relationship TEXT NOT NULL,
                name TEXT NOT NULL,
                birth_year INTEGER,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (politician_id) REFERENCES politicians (id)
            )
            """)
            
            # Create biography_sections table for long-form text
            conn.execute("""
            CREATE TABLE IF NOT EXISTS biography_sections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                politician_id INTEGER NOT NULL,
                section_title TEXT NOT NULL,
                section_content TEXT NOT NULL,
                source_url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (politician_id) REFERENCES politicians (id)
            )
            """)
            
            conn.commit()
    
    def add_politician(
        self, 
        name: str, 
        full_name: Optional[str] = None,
        birth_date: Optional[str] = None,
        birth_place: Optional[str] = None,
        party: Optional[str] = None,
        current_office: Optional[str] = None,
        image_url: Optional[str] = None,
        wikipedia_url: Optional[str] = None,
    ) -> int:
        """
        Add a politician to the database.
        
        Args:
            name: The politician's name
            full_name: The politician's full name
            birth_date: The politician's birth date (YYYY-MM-DD)
            birth_place: The politician's place of birth
            party: The politician's political party
            current_office: The politician's current office
            image_url: URL to the politician's image
            wikipedia_url: URL to the politician's Wikipedia page
            
        Returns:
            The ID of the newly created politician
        """
        query = """
        INSERT INTO politicians (
            name, full_name, birth_date, birth_place, party, 
            current_office, image_url, wikipedia_url
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (
                name, full_name, birth_date, birth_place, party,
                current_office, image_url, wikipedia_url
            ))
            conn.commit()
            return cursor.lastrowid
    
    def add_biographical_event(
        self,
        politician_id: int,
        event_date: Optional[str],
        event_type: str,
        event_description: str,
        source_url: Optional[str] = None,
    ) -> int:
        """
        Add a biographical event for a politician.
        
        Args:
            politician_id: The ID of the politician
            event_date: The date of the event (YYYY-MM-DD)
            event_type: The type of event (e.g., "birth", "marriage", "career")
            event_description: Description of the event
            source_url: URL to the source of the information
            
        Returns:
            The ID of the newly created event
        """
        query = """
        INSERT INTO biographical_events (
            politician_id, event_date, event_type, event_description, source_url
        ) VALUES (?, ?, ?, ?, ?)
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (
                politician_id, event_date, event_type, event_description, source_url
            ))
            conn.commit()
            return cursor.lastrowid
    
    def add_education(
        self,
        politician_id: int,
        institution: str,
        degree: Optional[str] = None,
        field_of_study: Optional[str] = None,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
    ) -> int:
        """
        Add education information for a politician.
        
        Args:
            politician_id: The ID of the politician
            institution: The educational institution
            degree: The degree earned
            field_of_study: The field of study
            start_year: The year the education started
            end_year: The year the education ended
            
        Returns:
            The ID of the newly created education record
        """
        query = """
        INSERT INTO education (
            politician_id, institution, degree, field_of_study, start_year, end_year
        ) VALUES (?, ?, ?, ?, ?, ?)
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (
                politician_id, institution, degree, field_of_study, start_year, end_year
            ))
            conn.commit()
            return cursor.lastrowid
    
    def add_career(
        self,
        politician_id: int,
        organization: str,
        role: str,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        description: Optional[str] = None,
    ) -> int:
        """
        Add career information for a politician.
        
        Args:
            politician_id: The ID of the politician
            organization: The organization or employer
            role: The role or position held
            start_year: The year the role started
            end_year: The year the role ended (or None if current)
            description: A description of the role
            
        Returns:
            The ID of the newly created career record
        """
        query = """
        INSERT INTO career (
            politician_id, organization, role, start_year, end_year, description
        ) VALUES (?, ?, ?, ?, ?, ?)
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (
                politician_id, organization, role, start_year, end_year, description
            ))
            conn.commit()
            return cursor.lastrowid
    
    def add_family_member(
        self,
        politician_id: int,
        relationship: str,
        name: str,
        birth_year: Optional[int] = None,
        description: Optional[str] = None,
    ) -> int:
        """
        Add a family member for a politician.
        
        Args:
            politician_id: The ID of the politician
            relationship: The relationship to the politician (e.g., "spouse", "child")
            name: The family member's name
            birth_year: The family member's birth year
            description: Additional description
            
        Returns:
            The ID of the newly created family record
        """
        query = """
        INSERT INTO family (
            politician_id, relationship, name, birth_year, description
        ) VALUES (?, ?, ?, ?, ?)
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (
                politician_id, relationship, name, birth_year, description
            ))
            conn.commit()
            return cursor.lastrowid
    
    def add_biography_section(
        self,
        politician_id: int,
        section_title: str,
        section_content: str,
        source_url: Optional[str] = None,
    ) -> int:
        """
        Add a biography section for a politician.
        
        Args:
            politician_id: The ID of the politician
            section_title: The title of the section
            section_content: The content of the section
            source_url: URL to the source of the information
            
        Returns:
            The ID of the newly created section
        """
        query = """
        INSERT INTO biography_sections (
            politician_id, section_title, section_content, source_url
        ) VALUES (?, ?, ?, ?)
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (
                politician_id, section_title, section_content, source_url
            ))
            conn.commit()
            
            # Add to embedding index
            self.embedding_index.add(
                [section_content],
                [{"politician_id": politician_id, "section_title": section_title, "source_url": source_url}]
            )
            self.embedding_index.save()
            
            return cursor.lastrowid
    
    def get_politician_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a politician by name.
        
        Args:
            name: The politician's name
            
        Returns:
            The politician's data or None if not found
        """
        query = "SELECT * FROM politicians WHERE name = ?"
        results = self.execute_query(query, (name,))
        return results[0] if results else None
    
    def get_politician_by_id(self, politician_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a politician by ID.
        
        Args:
            politician_id: The politician's ID
            
        Returns:
            The politician's data or None if not found
        """
        query = "SELECT * FROM politicians WHERE id = ?"
        results = self.execute_query(query, (politician_id,))
        return results[0] if results else None
    
    def get_biographical_events(self, politician_id: int) -> List[Dict[str, Any]]:
        """
        Get biographical events for a politician.
        
        Args:
            politician_id: The politician's ID
            
        Returns:
            A list of biographical events
        """
        query = "SELECT * FROM biographical_events WHERE politician_id = ? ORDER BY event_date"
        return self.execute_query(query, (politician_id,))
    
    def get_education(self, politician_id: int) -> List[Dict[str, Any]]:
        """
        Get education information for a politician.
        
        Args:
            politician_id: The politician's ID
            
        Returns:
            A list of education records
        """
        query = "SELECT * FROM education WHERE politician_id = ? ORDER BY start_year"
        return self.execute_query(query, (politician_id,))
    
    def get_career(self, politician_id: int) -> List[Dict[str, Any]]:
        """
        Get career information for a politician.
        
        Args:
            politician_id: The politician's ID
            
        Returns:
            A list of career records
        """
        query = "SELECT * FROM career WHERE politician_id = ? ORDER BY start_year DESC"
        return self.execute_query(query, (politician_id,))
    
    def get_family(self, politician_id: int) -> List[Dict[str, Any]]:
        """
        Get family information for a politician.
        
        Args:
            politician_id: The politician's ID
            
        Returns:
            A list of family records
        """
        query = "SELECT * FROM family WHERE politician_id = ?"
        return self.execute_query(query, (politician_id,))
    
    def get_biography_sections(self, politician_id: int) -> List[Dict[str, Any]]:
        """
        Get biography sections for a politician.
        
        Args:
            politician_id: The politician's ID
            
        Returns:
            A list of biography sections
        """
        query = "SELECT * FROM biography_sections WHERE politician_id = ?"
        return self.execute_query(query, (politician_id,))
    
    def search_biography(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the biography database for content related to the query.
        
        Args:
            query: The search query
            top_k: Number of results to return
            
        Returns:
            A list of search results
        """
        return self.embedding_index.search(query, top_k=top_k)
    
    def get_complete_biography(self, politician_id: int) -> Dict[str, Any]:
        """
        Get a complete biography for a politician.
        
        Args:
            politician_id: The politician's ID
            
        Returns:
            A dictionary with all biographical information
        """
        politician = self.get_politician_by_id(politician_id)
        if not politician:
            return {}
        
        return {
            "politician": politician,
            "events": self.get_biographical_events(politician_id),
            "education": self.get_education(politician_id),
            "career": self.get_career(politician_id),
            "family": self.get_family(politician_id),
            "sections": self.get_biography_sections(politician_id),
        }
    
    def import_from_wikipedia(self, name: str, wikipedia_title: Optional[str] = None) -> Optional[int]:
        """
        Import biographical information from Wikipedia.
        
        Args:
            name: The politician's name
            wikipedia_title: The Wikipedia page title (if different from name)
            
        Returns:
            The politician ID or None if import failed
        """
        # Use the provided Wikipedia title or the name
        title = wikipedia_title or name
        
        # Fetch data from Wikipedia
        wiki_data = fetch_wikipedia_data(title)
        
        # Extract page data
        pages = wiki_data.get("query", {}).get("pages", {})
        if not pages:
            return None
        
        # Get the first page (there should only be one)
        page = list(pages.values())[0]
        
        if "missing" in page:
            return None
        
        # Extract information
        full_name = name
        extract = page.get("extract", "")
        url = page.get("fullurl", "")
        
        # Try to extract birth date and place from the extract
        birth_date = None
        birth_place = None
        
        # Extract image URL if available
        image_url = None
        if "thumbnail" in page:
            image_url = page["thumbnail"].get("source")
        
        # Add the politician to the database
        politician_id = self.add_politician(
            name=name,
            full_name=full_name,
            birth_date=birth_date,
            birth_place=birth_place,
            wikipedia_url=url,
            image_url=image_url,
        )
        
        # Add the Wikipedia extract as a biography section
        if extract:
            self.add_biography_section(
                politician_id=politician_id,
                section_title="Wikipedia Overview",
                section_content=extract,
                source_url=url,
            )
        
        return politician_id