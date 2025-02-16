from typing import Dict, Optional, List
import json
from pathlib import Path
import time
from datetime import datetime, timedelta

class SessionManager:
    _instance = None
    _sessions: Dict[str, Dict] = {}
    _session_dir = Path.home() / ".cache" / "aipolitician" / "sessions"
    _max_session_age = timedelta(hours=24)  # Sessions expire after 24 hours

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SessionManager, cls).__new__(cls)
            cls._session_dir.mkdir(parents=True, exist_ok=True)
            cls._load_sessions()
        return cls._instance

    @classmethod
    def _load_sessions(cls):
        """Load existing sessions from disk."""
        for session_file in cls._session_dir.glob("*.json"):
            try:
                with open(session_file, "r") as f:
                    session_data = json.load(f)
                    # Check if session is still valid
                    last_access = datetime.fromtimestamp(session_data.get("last_access", 0))
                    if datetime.now() - last_access <= cls._max_session_age:
                        session_id = session_file.stem
                        cls._sessions[session_id] = session_data
                    else:
                        # Remove expired session file
                        session_file.unlink()
            except Exception as e:
                print(f"Error loading session {session_file}: {str(e)}")

    @classmethod
    def get_session(cls, session_id: str) -> Optional[Dict]:
        """Get session data by ID."""
        if session_id in cls._sessions:
            session_data = cls._sessions[session_id]
            # Update last access time
            session_data["last_access"] = time.time()
            cls._save_session(session_id, session_data)
            return session_data
        return None

    @classmethod
    def create_session(cls, session_id: str, agent_name: str) -> Dict:
        """Create a new session."""
        session_data = {
            "agent_name": agent_name,
            "messages": [],
            "created_at": time.time(),
            "last_access": time.time(),
            "metadata": {}
        }
        cls._sessions[session_id] = session_data
        cls._save_session(session_id, session_data)
        return session_data

    @classmethod
    def update_session(cls, session_id: str, messages: List[Dict], metadata: Optional[Dict] = None):
        """Update session with new messages and metadata."""
        if session_id in cls._sessions:
            session_data = cls._sessions[session_id]
            session_data["messages"] = messages
            if metadata:
                session_data["metadata"].update(metadata)
            session_data["last_access"] = time.time()
            cls._save_session(session_id, session_data)

    @classmethod
    def _save_session(cls, session_id: str, session_data: Dict):
        """Save session data to disk."""
        try:
            session_file = cls._session_dir / f"{session_id}.json"
            with open(session_file, "w") as f:
                json.dump(session_data, f)
        except Exception as e:
            print(f"Error saving session {session_id}: {str(e)}")

    @classmethod
    def cleanup_expired_sessions(cls):
        """Remove expired sessions."""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session_data in cls._sessions.items():
            last_access = datetime.fromtimestamp(session_data["last_access"])
            if current_time - last_access > cls._max_session_age:
                expired_sessions.append(session_id)
                # Remove session file
                session_file = cls._session_dir / f"{session_id}.json"
                if session_file.exists():
                    session_file.unlink()
        
        # Remove expired sessions from memory
        for session_id in expired_sessions:
            del cls._sessions[session_id]

    @classmethod
    def get_session_history(cls, session_id: str) -> List[Dict]:
        """Get conversation history for a session."""
        session = cls.get_session(session_id)
        return session["messages"] if session else []
