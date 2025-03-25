"""
Advanced RAG utilities for the political agent system.

Provides high-performance vector search and retrieval for political knowledge base.
"""

import os
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import json
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Vector database - determine best backend
try:
    # Try importing FAISS for best vector search performance
    import faiss
    USING_FAISS = True
    logger.info("Using FAISS backend for vector search")
except ImportError:
    try:
        # Fall back to ChromaDB
        import chromadb
        from chromadb.utils import embedding_functions
        USING_FAISS = False
        logger.info("Using ChromaDB backend for vector search")
    except ImportError:
        logger.warning("No vector database backend found. RAG will not function properly.")
        USING_FAISS = False

# Initialize sentence transformer for embeddings
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu")
    # Use half-precision for GPU efficiency
    if torch.cuda.is_available():
        EMBEDDING_MODEL = EMBEDDING_MODEL.half()
    logger.info(f"Embedding model initialized on {EMBEDDING_MODEL.device}")
except ImportError:
    logger.warning("SentenceTransformer not available. Using fallback embeddings.")
    EMBEDDING_MODEL = None

# Thread pool for parallel processing
MAX_WORKERS = min(os.cpu_count() or 4, 8)
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# Cache for embeddings to avoid recomputation
embedding_cache = {}
embedding_cache_limit = 10000  # Limit cache size to avoid memory issues


class FAISSKnowledgeBase:
    """FAISS-based knowledge base for efficient vector search."""
    
    def __init__(self, index_path: Optional[str] = None):
        """Initialize the FAISS knowledge base."""
        self.index = None
        self.documents = []
        self.persona_indices = {}  # Map persona to document indices
        
        if index_path and os.path.exists(index_path):
            self.load(index_path)
        else:
            # Initialize empty index
            self.index = faiss.IndexFlatIP(384)  # Default embedding dimension
    
    def add_documents(self, documents: List[Dict[str, Any]], persona: str) -> None:
        """Add documents to the knowledge base with persona association."""
        if not documents:
            return
        
        # Track existing document count
        start_idx = len(self.documents)
        
        # Extract texts and compute embeddings
        texts = [doc.get("text", "") for doc in documents]
        embeddings = compute_embeddings(texts)
        
        # Add embeddings to FAISS index
        if self.index is None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
        
        self.index.add(embeddings)
        
        # Store documents
        self.documents.extend(documents)
        
        # Update persona indices
        end_idx = len(self.documents)
        if persona not in self.persona_indices:
            self.persona_indices[persona] = []
        
        self.persona_indices[persona].extend(range(start_idx, end_idx))
    
    def search(self, query: str, persona: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents."""
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Compute query embedding
        query_embedding = compute_embeddings([query])
        
        # Search the entire index
        scores, indices = self.index.search(query_embedding, self.index.ntotal)
        
        # Filter by persona if specified
        if persona and persona in self.persona_indices:
            persona_doc_indices = set(self.persona_indices[persona])
            results = []
            
            for i, idx in enumerate(indices[0]):
                if idx < len(self.documents) and idx in persona_doc_indices:
                    doc = self.documents[idx].copy()
                    doc["score"] = float(scores[0][i])
                    results.append(doc)
                    
                    if len(results) >= top_k:
                        break
            
            return results
        else:
            # Return top_k results without persona filtering
            results = []
            for i, idx in enumerate(indices[0][:top_k]):
                if idx < len(self.documents):
                    doc = self.documents[idx].copy()
                    doc["score"] = float(scores[0][i])
                    results.append(doc)
            
            return results
    
    def save(self, index_path: str) -> bool:
        """Save the knowledge base to disk."""
        if self.index is None:
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, f"{index_path}.faiss")
            
            # Save documents and persona indices
            metadata = {
                "documents": self.documents,
                "persona_indices": self.persona_indices
            }
            
            with open(f"{index_path}.json", "w") as f:
                json.dump(metadata, f)
            
            return True
        except Exception as e:
            logger.error(f"Error saving knowledge base: {e}")
            return False
    
    def load(self, index_path: str) -> bool:
        """Load the knowledge base from disk."""
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{index_path}.faiss")
            
            # Load documents and persona indices
            with open(f"{index_path}.json", "r") as f:
                metadata = json.load(f)
                self.documents = metadata["documents"]
                self.persona_indices = metadata["persona_indices"]
            
            return True
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            self.index = faiss.IndexFlatIP(384)  # Create empty index
            self.documents = []
            self.persona_indices = {}
            return False


class ChromaKnowledgeBase:
    """ChromaDB-based knowledge base for vector search."""
    
    def __init__(self, collection_name: str = "political_knowledge"):
        """Initialize the ChromaDB knowledge base."""
        self.client = chromadb.Client()
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Create or get collection
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            self.collection = None
    
    def add_documents(self, documents: List[Dict[str, Any]], persona: str) -> None:
        """Add documents to the knowledge base with persona association."""
        if not documents or self.collection is None:
            return
        
        # Prepare data for ChromaDB
        ids = [f"{persona}-{i}-{int(time.time())}" for i in range(len(documents))]
        texts = [doc.get("text", "") for doc in documents]
        metadatas = [
            {
                "persona": persona,
                "source": doc.get("source", ""),
                "title": doc.get("title", ""),
                "date": doc.get("date", "")
            }
            for doc in documents
        ]
        
        # Add to collection
        try:
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
    
    def search(self, query: str, persona: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents."""
        if self.collection is None:
            return []
        
        try:
            # Prepare where clause for persona filtering
            where_clause = {"persona": persona} if persona else None
            
            # Search the collection
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_clause
            )
            
            # Format results
            documents = []
            for i, doc in enumerate(results.get("documents", [[]])[0]):
                if i < len(results.get("metadatas", [[]])[0]) and i < len(results.get("distances", [[]])[0]):
                    metadata = results["metadatas"][0][i]
                    score = 1.0 - results["distances"][0][i]  # Convert distance to similarity score
                    
                    documents.append({
                        "text": doc,
                        "score": score,
                        "source": metadata.get("source", ""),
                        "title": metadata.get("title", ""),
                        "date": metadata.get("date", "")
                    })
            
            return documents
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}")
            return []


def compute_embeddings(texts: List[str]) -> np.ndarray:
    """Compute embeddings for a list of texts with caching."""
    if not EMBEDDING_MODEL:
        # Fallback simple embedding method
        return np.random.rand(len(texts), 384).astype(np.float32)
    
    # Check cache first
    uncached_texts = []
    uncached_indices = []
    embeddings = np.zeros((len(texts), 384), dtype=np.float32)
    
    for i, text in enumerate(texts):
        if text in embedding_cache:
            embeddings[i] = embedding_cache[text]
        else:
            uncached_texts.append(text)
            uncached_indices.append(i)
    
    # Compute embeddings for uncached texts
    if uncached_texts:
        with torch.no_grad():
            new_embeddings = EMBEDDING_MODEL.encode(
                uncached_texts,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        
        # Update cache and embeddings array
        for i, idx in enumerate(uncached_indices):
            embedding = new_embeddings[i]
            text = uncached_texts[i]
            
            # Add to cache if not full
            if len(embedding_cache) < embedding_cache_limit:
                embedding_cache[text] = embedding
            
            embeddings[idx] = embedding
    
    return embeddings


# Initialize the knowledge base
if USING_FAISS:
    KB = FAISSKnowledgeBase(index_path="db/data/knowledge_base")
else:
    KB = ChromaKnowledgeBase(collection_name="political_knowledge")


def enhance_query(query: str, persona: str) -> str:
    """Enhance the query with contextual information for better retrieval."""
    if not query:
        return query
    
    # Add persona-specific context
    enhanced = f"In context of {persona}'s political stance: {query}"
    
    return enhanced


def integrate_with_chat(
    query: str, 
    persona: str, 
    top_k: int = 8, 
    enable_reranking: bool = True
) -> Tuple[str, bool]:
    """
    Integrate RAG with the chat system.
    
    Args:
        query: The user query
        persona: The politician persona
        top_k: Number of documents to retrieve
        enable_reranking: Whether to rerank results
    
    Returns:
        A tuple of (context_text, success)
    """
    if not query or not KB:
        return "", False
    
    try:
        # Search for relevant documents
        start_time = time.time()
        results = KB.search(query, persona, top_k=top_k + 4)  # Get extra for reranking
        
        if not results:
            return "", False
        
        # Rerank results if enabled
        if enable_reranking and len(results) > top_k:
            results = rerank_results(query, results, top_k)
        
        # Format results for context
        context_parts = []
        for i, doc in enumerate(results[:top_k]):
            text = doc.get("text", "").strip()
            source = doc.get("source", "unknown source")
            
            if text:
                context_parts.append(f"[{i+1}] {text} (Source: {source})")
        
        context = "\n\n".join(context_parts)
        
        # Log timing
        elapsed = time.time() - start_time
        logger.info(f"RAG retrieval completed in {elapsed:.3f}s")
        
        return context, True
    except Exception as e:
        logger.error(f"Error in RAG integration: {e}")
        return "", False


def rerank_results(query: str, results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """Rerank search results for better relevance."""
    if len(results) <= top_k:
        return results
    
    try:
        # Simple cross-encoder style reranking using dot product similarity
        query_embedding = compute_embeddings([query])[0]
        
        # Extract texts
        texts = [doc.get("text", "") for doc in results]
        
        # Compute fresh embeddings for cross-attention
        doc_embeddings = compute_embeddings(texts)
        
        # Calculate cross-attention scores
        similarities = np.dot(doc_embeddings, query_embedding)
        
        # Sort by similarity
        reranked_indices = np.argsort(-similarities)
        
        # Rerank results
        reranked_results = [results[idx] for idx in reranked_indices[:top_k]]
        
        return reranked_results
    except Exception as e:
        logger.error(f"Error in reranking: {e}")
        return results[:top_k]


def add_documents_from_path(
    path: str, 
    persona: str,
    file_ext: str = ".txt",
    batch_size: int = 100
) -> int:
    """
    Add documents from a directory to the knowledge base.
    
    Args:
        path: Path to directory containing documents
        persona: Persona to associate with documents
        file_ext: File extension to filter by
        batch_size: Number of documents to process in one batch
    
    Returns:
        Number of documents added
    """
    if not os.path.exists(path):
        logger.error(f"Path does not exist: {path}")
        return 0
    
    try:
        # Collect all matching files
        files = []
        for root, _, filenames in os.walk(path):
            for filename in filenames:
                if filename.endswith(file_ext):
                    files.append(os.path.join(root, filename))
        
        # Process files in batches
        total_docs = 0
        for i in range(0, len(files), batch_size):
            batch_files = files[i:i+batch_size]
            batch_docs = []
            
            # Process batch in parallel
            def process_file(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                        
                    if not text:
                        return None
                    
                    return {
                        "text": text,
                        "source": os.path.basename(file_path),
                        "title": os.path.splitext(os.path.basename(file_path))[0],
                        "date": ""
                    }
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    return None
            
            # Process batch in parallel
            results = list(executor.map(process_file, batch_files))
            batch_docs = [doc for doc in results if doc is not None]
            
            # Add batch to knowledge base
            if batch_docs:
                KB.add_documents(batch_docs, persona)
                total_docs += len(batch_docs)
                logger.info(f"Added {len(batch_docs)} documents to knowledge base")
        
        return total_docs
    except Exception as e:
        logger.error(f"Error adding documents from path: {e}")
        return 0


def save_knowledge_base() -> bool:
    """Save the knowledge base to disk."""
    if USING_FAISS and isinstance(KB, FAISSKnowledgeBase):
        return KB.save("db/data/knowledge_base")
    return False

