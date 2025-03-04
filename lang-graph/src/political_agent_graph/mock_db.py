"""Mock database for storing conversation history.

This is a simple in-memory database for demonstration purposes.
In a production environment, this would be replaced with a real database.
"""

from typing import Dict, List, Any

# Simple in-memory database
_db = {
    "conversations": {}
}

def add_conversation(user_id: str, conversation_data: Dict[str, Any]) -> str:
    """Add a new conversation to the database.
    
    Args:
        user_id: The ID of the user
        conversation_data: The conversation data to store
        
    Returns:
        The ID of the conversation
    """
    conversation_id = f"conv_{len(_db['conversations']) + 1}"
    _db["conversations"][conversation_id] = {
        "user_id": user_id,
        "messages": [conversation_data],
        "metadata": {
            "created_at": "2025-03-04T12:00:00Z",  # Placeholder timestamp
            "politician_id": conversation_data.get("politician_id", "unknown")
        }
    }
    return conversation_id

def add_message(conversation_id: str, message: Dict[str, Any]) -> bool:
    """Add a message to an existing conversation.
    
    Args:
        conversation_id: The ID of the conversation
        message: The message to add
        
    Returns:
        True if successful, False otherwise
    """
    if conversation_id in _db["conversations"]:
        _db["conversations"][conversation_id]["messages"].append(message)
        return True
    return False

def get_conversation(conversation_id: str) -> Dict[str, Any]:
    """Get a conversation by ID.
    
    Args:
        conversation_id: The ID of the conversation
        
    Returns:
        The conversation data, or None if not found
    """
    return _db["conversations"].get(conversation_id)

def get_user_conversations(user_id: str) -> List[Dict[str, Any]]:
    """Get all conversations for a user.
    
    Args:
        user_id: The ID of the user
        
    Returns:
        A list of conversation data
    """
    return [
        {"id": conv_id, **conv_data} 
        for conv_id, conv_data in _db["conversations"].items()
        if conv_data["user_id"] == user_id
    ]

def clear_all():
    """Clear all data from the database."""
    _db["conversations"] = {}