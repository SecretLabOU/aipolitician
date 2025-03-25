"""
Political Agent Database Module

High-performance vector database and knowledge integration system.
"""

__version__ = "1.0.0"

# Expose main API
from db.utils import (
    integrate_with_chat,
    enhance_query,
    add_documents_from_path,
    save_knowledge_base,
)

__all__ = [
    "integrate_with_chat",
    "enhance_query", 
    "add_documents_from_path",
    "save_knowledge_base",
    "__version__",
] 