from uuid import uuid4
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import threading
import logging

@dataclass
class Session:
    history: List[tuple[str, str]] = field(default_factory=list)
    last_access: datetime = field(default_factory=datetime.now)
    agent: Optional[object] = None
    
    def add_interaction(self, user_msg: str, agent_msg: str):
        """Add a new interaction to the conversation history"""
        self.history.append((user_msg, agent_msg))
        self.last_access = datetime.now()
    
    def cleanup(self):
        """Clean up session resources"""
        if self.agent and hasattr(self.agent, 'cleanup'):
            try:
                self.agent.cleanup()
            except Exception as e:
                logging.error(f"Error cleaning up agent: {str(e)}")
        self.agent = None

class SessionManager:
    def __init__(self, session_timeout_minutes: int = 30):
        self.sessions: Dict[str, Session] = {}
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self.cleanup_lock = threading.Lock()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    def get_session(self, session_id: str = "default") -> Session:
        """Get or create a session"""
        if session_id == "default":
            session_id = str(uuid4())
            
        with self.cleanup_lock:
            session = self.sessions.get(session_id)
            if not session:
                session = Session()
                self.sessions[session_id] = session
            session.last_access = datetime.now()
            
        return session
    
    def set_agent_for_session(self, session_id: str, agent: object):
        """Set the agent for a session"""
        with self.cleanup_lock:
            session = self.get_session(session_id)
            if session.agent:
                session.cleanup()
            session.agent = agent
    
    def cleanup_session(self, session_id: str):
        """Clean up a specific session"""
        with self.cleanup_lock:
            session = self.sessions.get(session_id)
            if session:
                session.cleanup()
                del self.sessions[session_id]
    
    def _cleanup_loop(self):
        """Background task to clean up inactive sessions"""
        while True:
            try:
                self._cleanup_inactive_sessions()
            except Exception as e:
                logging.error(f"Error in cleanup loop: {str(e)}")
            threading.Event().wait(300)  # Run every 5 minutes
    
    def _cleanup_inactive_sessions(self):
        """Clean up sessions that have been inactive for too long"""
        current_time = datetime.now()
        with self.cleanup_lock:
            expired_sessions = [
                session_id for session_id, session in self.sessions.items()
                if current_time - session.last_access > self.session_timeout
            ]
            for session_id in expired_sessions:
                self.cleanup_session(session_id)
