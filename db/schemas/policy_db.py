"""
Policy Database Schema.

This module defines the schema for the Policy Database, which stores
information about politician's policy positions and platforms.
"""
from typing import Dict, List, Optional, Any, Tuple

from db.database import Database
from db.utils.embedding_utils import get_embedding_index


class PolicyDatabase(Database):
    """Policy Database for storing policy positions and platforms."""
    
    def __init__(self, db_name: str = "policy"):
        """
        Initialize the Policy Database.
        
        Args:
            db_name: The name of the database
        """
        super().__init__(db_name)
        self.embedding_index = get_embedding_index("policy")
    
    def initialize_db(self) -> None:
        """Initialize the database with the required schema."""
        with self.get_connection() as conn:
            # Create policies table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS policies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                politician_id INTEGER NOT NULL,
                policy_area TEXT NOT NULL,
                policy_position TEXT NOT NULL,
                summary TEXT NOT NULL,
                source_url TEXT,
                source_date TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (politician_id) REFERENCES politicians (id)
            )
            """)
            
            # Create policy_details table for more detailed information
            conn.execute("""
            CREATE TABLE IF NOT EXISTS policy_details (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                policy_id INTEGER NOT NULL,
                detail_type TEXT NOT NULL,
                detail_content TEXT NOT NULL,
                source_url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (policy_id) REFERENCES policies (id)
            )
            """)
            
            conn.commit()
    
    def add_policy(
        self, 
        politician_id: int,
        policy_area: str,
        policy_position: str,
        summary: str,
        source_url: Optional[str] = None,
        source_date: Optional[str] = None,
    ) -> int:
        """
        Add a policy position for a politician.
        
        Args:
            politician_id: The ID of the politician
            policy_area: The area of policy (e.g., "Healthcare", "Education")
            policy_position: A short statement of the position
            summary: A summary of the policy position
            source_url: URL to the source of the information
            source_date: Date the position was stated or published
            
        Returns:
            The ID of the newly created policy
        """
        query = """
        INSERT INTO policies (
            politician_id, policy_area, policy_position, summary, source_url, source_date
        ) VALUES (?, ?, ?, ?, ?, ?)
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (
                politician_id, policy_area, policy_position, summary, source_url, source_date
            ))
            conn.commit()
            
            # Add to embedding index
            self.embedding_index.add(
                [f"{policy_area}: {policy_position}\n{summary}"],
                [{"policy_id": cursor.lastrowid, "policy_area": policy_area}]
            )
            self.embedding_index.save()
            
            return cursor.lastrowid
    
    def add_policy_detail(
        self,
        policy_id: int,
        detail_type: str,
        detail_content: str,
        source_url: Optional[str] = None,
    ) -> int:
        """
        Add a detail to a policy position.
        
        Args:
            policy_id: The ID of the policy
            detail_type: The type of detail (e.g., "proposal", "critique", "implementation")
            detail_content: The content of the detail
            source_url: URL to the source of the information
            
        Returns:
            The ID of the newly created detail
        """
        query = """
        INSERT INTO policy_details (
            policy_id, detail_type, detail_content, source_url
        ) VALUES (?, ?, ?, ?)
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (
                policy_id, detail_type, detail_content, source_url
            ))
            conn.commit()
            
            # Add to embedding index
            policy = self.get_policy_by_id(policy_id)
            if policy:
                self.embedding_index.add(
                    [detail_content],
                    [{
                        "policy_id": policy_id, 
                        "policy_area": policy["policy_area"],
                        "detail_type": detail_type
                    }]
                )
                self.embedding_index.save()
            
            return cursor.lastrowid
    
    def get_policy_by_id(self, policy_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a policy by ID.
        
        Args:
            policy_id: The policy's ID
            
        Returns:
            The policy's data or None if not found
        """
        query = "SELECT * FROM policies WHERE id = ?"
        results = self.execute_query(query, (policy_id,))
        return results[0] if results else None
    
    def get_policies_by_politician(self, politician_id: int) -> List[Dict[str, Any]]:
        """
        Get policies for a politician.
        
        Args:
            politician_id: The politician's ID
            
        Returns:
            A list of policies
        """
        query = "SELECT * FROM policies WHERE politician_id = ? ORDER BY policy_area"
        return self.execute_query(query, (politician_id,))
    
    def get_policies_by_area(self, policy_area: str) -> List[Dict[str, Any]]:
        """
        Get policies for a specific policy area.
        
        Args:
            policy_area: The policy area
            
        Returns:
            A list of policies
        """
        query = "SELECT * FROM policies WHERE policy_area = ?"
        return self.execute_query(query, (policy_area,))
    
    def get_policy_details(self, policy_id: int) -> List[Dict[str, Any]]:
        """
        Get details for a policy.
        
        Args:
            policy_id: The policy's ID
            
        Returns:
            A list of policy details
        """
        query = "SELECT * FROM policy_details WHERE policy_id = ?"
        return self.execute_query(query, (policy_id,))
    
    def search_policies(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the policy database for content related to the query.
        
        Args:
            query: The search query
            top_k: Number of results to return
            
        Returns:
            A list of search results
        """
        return self.embedding_index.search(query, top_k=top_k)
