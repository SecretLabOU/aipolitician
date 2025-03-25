"""
Database utilities for the Political Agent system.

Provides high-performance vector storage and retrieval capabilities.
"""

__version__ = "1.0.0"

from .rag_utils import (
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