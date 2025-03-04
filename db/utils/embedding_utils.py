"""
Utility functions for text embedding and retrieval.

This module provides functionality for embedding text and retrieving
similar text for the Political RAG system.
"""
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# In a production system, you would use a more sophisticated embedding model
# such as sentence-transformers, HuggingFace Transformers, or OpenAI's API
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
except ImportError:
    EMBEDDING_MODEL = None
    print("Warning: sentence-transformers not installed. Using dummy embedding model.")


class DummyEmbeddingModel:
    """A dummy embedding model for demonstration purposes."""
    
    def encode(self, texts: Union[str, List[str]], **kwargs):
        """
        Generate dummy embeddings.
        
        Args:
            texts: Text or list of texts to encode
            
        Returns:
            Numpy array of dummy embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Generate deterministic but unique embeddings based on text content
        # This is just for demonstration and should not be used in production
        return np.array([
            np.array([hash(text) % 10000 for _ in range(384)]) / 10000
            for text in texts
        ])


def get_embedding_model():
    """
    Get the embedding model.
    
    Returns:
        An embedding model that implements the encode method
    """
    return EMBEDDING_MODEL or DummyEmbeddingModel()


def embed_text(text: str) -> np.ndarray:
    """
    Embed text using the embedding model.
    
    Args:
        text: The text to embed
        
    Returns:
        A vector embedding of the text
    """
    model = get_embedding_model()
    return model.encode(text)


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed multiple texts using the embedding model.
    
    Args:
        texts: The texts to embed
        
    Returns:
        A matrix of text embeddings
    """
    model = get_embedding_model()
    return model.encode(texts)


def find_similar_texts(
    query: str, 
    texts: List[str], 
    embeddings: Optional[np.ndarray] = None,
    top_k: int = 5
) -> List[Tuple[int, float, str]]:
    """
    Find texts similar to the query text.
    
    Args:
        query: The query text
        texts: The candidate texts
        embeddings: Pre-computed embeddings for the candidate texts
        top_k: Number of similar texts to return
        
    Returns:
        A list of (index, similarity_score, text) tuples
    """
    # Embed the query
    query_embedding = embed_text(query).reshape(1, -1)
    
    # Embed the texts if embeddings are not provided
    if embeddings is None:
        embeddings = embed_texts(texts)
    
    # Calculate similarity scores
    similarity_scores = cosine_similarity(query_embedding, embeddings)[0]
    
    # Get the indices of the top_k most similar texts
    top_indices = np.argsort(similarity_scores)[-top_k:][::-1]
    
    # Return the top_k most similar texts with their similarity scores
    return [(idx, similarity_scores[idx], texts[idx]) for idx in top_indices]


class EmbeddingIndex:
    """A simple in-memory embedding index for demonstration purposes."""
    
    def __init__(self, name: str):
        """
        Initialize the embedding index.
        
        Args:
            name: The name of the index
        """
        self.name = name
        self.texts = []
        self.metadata = []
        self.embeddings = None
        self.index_path = Path('/home/natalie/Databases/political_rag') / f"{name}_embeddings.pkl"
    
    def add(
        self, 
        texts: List[str], 
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Add texts to the index.
        
        Args:
            texts: The texts to add
            metadata: Metadata for each text
        """
        if not texts:
            return
            
        if metadata is None:
            metadata = [{} for _ in texts]
        
        # Generate embeddings for the new texts
        new_embeddings = embed_texts(texts)
        
        # Add the new texts, metadata, and embeddings to the index
        self.texts.extend(texts)
        self.metadata.extend(metadata)
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the index for texts similar to the query.
        
        Args:
            query: The query text
            top_k: Number of similar texts to return
            
        Returns:
            A list of search results with text, metadata, and score
        """
        if not self.texts:
            return []
        
        # Find similar texts
        similar_texts = find_similar_texts(
            query, 
            self.texts, 
            self.embeddings, 
            top_k=top_k
        )
        
        # Convert to search results
        return [
            {
                "text": text,
                "metadata": self.metadata[idx],
                "score": score
            }
            for idx, score, text in similar_texts
        ]
    
    def save(self) -> None:
        """Save the embedding index to disk."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # Save the index
        with open(self.index_path, 'wb') as f:
            pickle.dump({
                'texts': self.texts,
                'metadata': self.metadata,
                'embeddings': self.embeddings
            }, f)
    
    def load(self) -> bool:
        """
        Load the embedding index from disk.
        
        Returns:
            True if the index was loaded successfully, False otherwise
        """
        if not os.path.exists(self.index_path):
            return False
        
        # Load the index
        with open(self.index_path, 'rb') as f:
            data = pickle.load(f)
            
        self.texts = data['texts']
        self.metadata = data['metadata']
        self.embeddings = data['embeddings']
        
        return True


def get_embedding_index(name: str) -> EmbeddingIndex:
    """
    Get an embedding index.
    
    Args:
        name: The name of the index
        
    Returns:
        An EmbeddingIndex
    """
    index = EmbeddingIndex(name)
    index.load()  # Try to load existing index
    return index