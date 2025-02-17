from uuid import uuid4
from collections import defaultdict
from dataclasses import dataclass, field

@dataclass
class Session:
    history: list[tuple[str, str]] = field(default_factory=list)
    
    def add_interaction(self, user_msg: str, agent_msg: str):
        self.history.append((user_msg, agent_msg))

class SessionManager:
    def __init__(self):
        self.sessions = defaultdict(Session)
    
    def get_session(self, session_id: str) -> Session:
        if session_id == "default":
            session_id = str(uuid4())
        return self.sessions[session_id]