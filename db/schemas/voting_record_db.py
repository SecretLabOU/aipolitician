"""
Voting Record Database Schema.

This module defines the schema for the Voting Record Database, which stores
information about politicians' voting history on bills and resolutions.
"""
from typing import Dict, List, Optional, Any, Tuple

from db.database import Database
from db.utils.embedding_utils import get_embedding_index


class VotingRecordDatabase(Database):
    """Voting Record Database for storing voting history."""
    
    def __init__(self, db_name: str = "voting_record"):
        """
        Initialize the Voting Record Database.
        
        Args:
            db_name: The name of the database
        """
        super().__init__(db_name)
        self.embedding_index = get_embedding_index("voting_record")
    
    def initialize_db(self) -> None:
        """Initialize the database with the required schema."""
        with self.get_connection() as conn:
            # Create bills table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS bills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bill_number TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                chamber TEXT NOT NULL,
                introduced_date TEXT,
                status TEXT,
                url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # Create votes table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS votes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                politician_id INTEGER NOT NULL,
                bill_id INTEGER NOT NULL,
                vote TEXT NOT NULL,
                vote_date TEXT,
                explanation TEXT,
                source_url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (politician_id) REFERENCES politicians (id),
                FOREIGN KEY (bill_id) REFERENCES bills (id)
            )
            """)
            
            # Create bill_topics table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS bill_topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bill_id INTEGER NOT NULL,
                topic TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (bill_id) REFERENCES bills (id)
            )
            """)
            
            conn.commit()
    
    def add_bill(
        self,
        bill_number: str,
        title: str,
        description: Optional[str] = None,
        chamber: str = "house",
        introduced_date: Optional[str] = None,
        status: Optional[str] = None,
        url: Optional[str] = None,
    ) -> int:
        """
        Add a bill to the database.
        
        Args:
            bill_number: The bill's number (e.g., "HR 1234")
            title: The bill's title
            description: A description of the bill
            chamber: The chamber (house or senate)
            introduced_date: The date the bill was introduced (YYYY-MM-DD)
            status: The current status of the bill
            url: URL to more information about the bill
            
        Returns:
            The ID of the newly created bill
        """
        query = """
        INSERT INTO bills (
            bill_number, title, description, chamber, introduced_date, status, url
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (
                bill_number, title, description, chamber, introduced_date, status, url
            ))
            conn.commit()
            
            bill_id = cursor.lastrowid
            
            # Add to embedding index
            bill_text = f"{bill_number}: {title}"
            if description:
                bill_text += f"\n{description}"
            
            self.embedding_index.add(
                [bill_text],
                [{
                    "bill_id": bill_id,
                    "bill_number": bill_number,
                    "title": title
                }]
            )
            self.embedding_index.save()
            
            return bill_id
    
    def add_vote(
        self,
        politician_id: int,
        bill_id: int,
        vote: str,
        vote_date: Optional[str] = None,
        explanation: Optional[str] = None,
        source_url: Optional[str] = None,
    ) -> int:
        """
        Add a politician's vote on a bill.
        
        Args:
            politician_id: The ID of the politician
            bill_id: The ID of the bill
            vote: The vote (e.g., "yes", "no", "abstain")
            vote_date: The date of the vote (YYYY-MM-DD)
            explanation: Explanation or statement about the vote
            source_url: URL to the source of the information
            
        Returns:
            The ID of the newly created vote
        """
        query = """
        INSERT INTO votes (
            politician_id, bill_id, vote, vote_date, explanation, source_url
        ) VALUES (?, ?, ?, ?, ?, ?)
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (
                politician_id, bill_id, vote, vote_date, explanation, source_url
            ))
            conn.commit()
            
            # Add to embedding index if there's an explanation
            if explanation:
                bill = self.get_bill_by_id(bill_id)
                if bill:
                    self.embedding_index.add(
                        [explanation],
                        [{
                            "vote_id": cursor.lastrowid,
                            "bill_id": bill_id,
                            "bill_number": bill["bill_number"],
                            "vote": vote
                        }]
                    )
                    self.embedding_index.save()
            
            return cursor.lastrowid
    
    def add_bill_topic(
        self,
        bill_id: int,
        topic: str,
    ) -> int:
        """
        Add a topic for a bill.
        
        Args:
            bill_id: The ID of the bill
            topic: The topic
            
        Returns:
            The ID of the newly created bill topic
        """
        query = """
        INSERT INTO bill_topics (
            bill_id, topic
        ) VALUES (?, ?)
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (bill_id, topic))
            conn.commit()
            return cursor.lastrowid
    
    def get_bill_by_id(self, bill_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a bill by ID.
        
        Args:
            bill_id: The bill's ID
            
        Returns:
            The bill's data or None if not found
        """
        query = "SELECT * FROM bills WHERE id = ?"
        results = self.execute_query(query, (bill_id,))
        return results[0] if results else None
    
    def get_bill_by_number(self, bill_number: str) -> Optional[Dict[str, Any]]:
        """
        Get a bill by number.
        
        Args:
            bill_number: The bill's number (e.g., "HR 1234")
            
        Returns:
            The bill's data or None if not found
        """
        query = "SELECT * FROM bills WHERE bill_number = ?"
        results = self.execute_query(query, (bill_number,))
        return results[0] if results else None
    
    def get_votes_by_politician(self, politician_id: int) -> List[Dict[str, Any]]:
        """
        Get votes for a politician.
        
        Args:
            politician_id: The politician's ID
            
        Returns:
            A list of votes
        """
        query = """
        SELECT v.*, b.bill_number, b.title, b.chamber
        FROM votes v
        JOIN bills b ON v.bill_id = b.id
        WHERE v.politician_id = ?
        ORDER BY v.vote_date DESC
        """
        return self.execute_query(query, (politician_id,))
    
    def get_votes_by_bill(self, bill_id: int) -> List[Dict[str, Any]]:
        """
        Get votes for a bill.
        
        Args:
            bill_id: The bill's ID
            
        Returns:
            A list of votes
        """
        query = "SELECT * FROM votes WHERE bill_id = ?"
        return self.execute_query(query, (bill_id,))
    
    def get_bills_by_topic(self, topic: str) -> List[Dict[str, Any]]:
        """
        Get bills for a topic.
        
        Args:
            topic: The topic
            
        Returns:
            A list of bills
        """
        query = """
        SELECT b.*
        FROM bills b
        JOIN bill_topics bt ON b.id = bt.bill_id
        WHERE bt.topic = ?
        """
        return self.execute_query(query, (topic,))
    
    def get_bill_topics(self, bill_id: int) -> List[str]:
        """
        Get topics for a bill.
        
        Args:
            bill_id: The bill's ID
            
        Returns:
            A list of topics
        """
        query = "SELECT topic FROM bill_topics WHERE bill_id = ?"
        results = self.execute_query(query, (bill_id,))
        return [result["topic"] for result in results]
    
    def search_voting_records(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the voting record database for content related to the query.
        
        Args:
            query: The search query
            top_k: Number of results to return
            
        Returns:
            A list of search results
        """
        return self.embedding_index.search(query, top_k=top_k)
