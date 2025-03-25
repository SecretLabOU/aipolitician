#!/usr/bin/env python3
"""
Advanced Hybrid RAG Engine

High-performance retrieval system with multiple vector stores, knowledge fusion,
and adaptive relevance scoring for maximum accuracy and performance.
"""

import os
import logging
import time
import asyncio
import hashlib
import json
from typing import Dict, List, Optional, Any, Union, Tuple, Set, AsyncGenerator
from pathlib import Path
from functools import lru_cache
from datetime import datetime
import concurrent.futures
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger("rag_system")

# Load environment variables
load_dotenv()

# Vector Store Configuration
VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "milvus").lower()
ENABLE_HYBRID_SEARCH = os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() == "true"
USE_QUERY_EXPANSION = os.getenv("USE_QUERY_EXPANSION", "true").lower() == "true"
LOCAL_CACHE_SIZE = int(os.getenv("RAG_CACHE_SIZE", "1000"))
ENABLE_KNOWLEDGE_FUSION = os.getenv("ENABLE_KNOWLEDGE_FUSION", "true").lower() == "true"

# Milvus specific configuration
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
MILVUS_USER = os.getenv("MILVUS_USER", "")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "political_facts")
MILVUS_CONSISTENCY_LEVEL = os.getenv("MILVUS_CONSISTENCY_LEVEL", "Session")
EMBED_DIM = int(os.getenv("EMBEDDING_DIMENSION", "768"))

# Embedding model configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
MULTI_EMBEDDING_MODELS = os.getenv("MULTI_EMBEDDING_MODELS", "").split(",")
ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "true").lower() == "true"

# Performance settings
NUM_RESULTS = int(os.getenv("RAG_NUM_RESULTS", "5"))
RAG_TIMEOUT = float(os.getenv("RAG_TIMEOUT", "3.0"))
RETRIEVAL_THREADS = int(os.getenv("RETRIEVAL_THREADS", "4"))
USE_PARALLEL_EMBEDDING = os.getenv("USE_PARALLEL_EMBEDDING", "true").lower() == "true"

# Import optional dependencies
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("Sentence-transformers not installed. Using fallback embedding methods.")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Setup vector store imports based on configuration
if VECTOR_STORE_TYPE == "milvus":
    try:
        from pymilvus import (
            connections, 
            utility,
            Collection,
            CollectionSchema,
            FieldSchema,
            DataType,
            SearchResult,
            Hit,
            ConsistencyLevel
        )
        MILVUS_AVAILABLE = True
    except ImportError:
        logger.warning("Pymilvus not installed. Using fallback retrieval methods.")
        MILVUS_AVAILABLE = False
elif VECTOR_STORE_TYPE == "faiss":
    try:
        import faiss
        import numpy as np
        FAISS_AVAILABLE = True
    except ImportError:
        logger.warning("FAISS not installed. Using fallback retrieval methods.")
        FAISS_AVAILABLE = False
else:
    logger.warning(f"Unknown vector store type: {VECTOR_STORE_TYPE}. Using fallback methods.")

# Cache for query results
class QueryCache:
    """Fast in-memory cache for query results with LRU eviction"""
    
    def __init__(self, max_size=LOCAL_CACHE_SIZE):
        self.max_size = max_size
        self.cache = {}
        self.usage_order = []
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if present and update usage order"""
        if key in self.cache:
            # Update usage order
            self.usage_order.remove(key)
            self.usage_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: Any) -> None:
        """Add value to cache with LRU eviction if needed"""
        # Evict oldest entry if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest_key = self.usage_order.pop(0)
            del self.cache[oldest_key]
        
        # Add new entry
        self.cache[key] = value
        
        # Update usage order
        if key in self.usage_order:
            self.usage_order.remove(key)
        self.usage_order.append(key)
    
    def clear(self) -> None:
        """Clear the cache completely"""
        self.cache = {}
        self.usage_order = []

class HybridRAGEngine:
    """Advanced RAG system with hybrid retrieval and knowledge fusion"""
    
    def __init__(self):
        """Initialize the advanced RAG engine"""
        self.embedding_models = {}
        self.reranker = None
        self.vector_store = None
        self.connected = False
        self.query_cache = QueryCache()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=RETRIEVAL_THREADS)
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all system components"""
        # Initialize embedding models
        self._initialize_embedding_models()
        
        # Initialize reranker if enabled
        if ENABLE_RERANKING and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                reranker_model = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
                self.reranker = CrossEncoder(reranker_model)
                logger.info(f"Loaded reranker model: {reranker_model}")
            except Exception as e:
                logger.error(f"Failed to load reranker model: {str(e)}")
        
        # Initialize vector store
        self._initialize_vector_store()
    
    def _initialize_embedding_models(self):
        """Initialize embedding models for vectorization"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("Sentence-transformers not available, embeddings will be limited")
            return
        
        try:
            # Load primary embedding model
            self.embedding_models["default"] = SentenceTransformer(EMBEDDING_MODEL)
            logger.info(f"Loaded primary embedding model: {EMBEDDING_MODEL}")
            
            # Load additional embedding models if specified
            if MULTI_EMBEDDING_MODELS and ENABLE_HYBRID_SEARCH:
                for model_name in MULTI_EMBEDDING_MODELS:
                    if model_name and model_name != EMBEDDING_MODEL:
                        self.embedding_models[model_name] = SentenceTransformer(model_name)
                        logger.info(f"Loaded additional embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding models: {str(e)}")
    
    def _initialize_vector_store(self):
        """Initialize the vector store based on configuration"""
        if VECTOR_STORE_TYPE == "milvus" and MILVUS_AVAILABLE:
            self._connect_to_milvus()
        elif VECTOR_STORE_TYPE == "faiss" and FAISS_AVAILABLE:
            self._initialize_faiss()
        else:
            logger.warning("No vector store available. Using fallback retrieval methods.")
    
    def _connect_to_milvus(self):
        """Connect to Milvus vector database with optimized settings"""
        try:
            # Connect to Milvus server
            connections.connect(
                alias="default",
                host=MILVUS_HOST,
                port=MILVUS_PORT,
                user=MILVUS_USER or None,
                password=MILVUS_PASSWORD or None,
                timeout=30
            )
            
            # Get consistency level enum value
            consistency_level = getattr(ConsistencyLevel, MILVUS_CONSISTENCY_LEVEL, ConsistencyLevel.Session)
            
            # Check if collection exists
            if utility.has_collection(COLLECTION_NAME):
                self.vector_store = Collection(COLLECTION_NAME)
                self.vector_store.load()
                # Set consistency level for better performance
                self.vector_store.consistency_level = consistency_level
                logger.info(f"Connected to Milvus collection: {COLLECTION_NAME}")
                self.connected = True
            else:
                logger.warning(f"Collection {COLLECTION_NAME} does not exist in Milvus.")
        except Exception as e:
            logger.error(f"Error connecting to Milvus: {str(e)}")
            self.connected = False
    
    def _initialize_faiss(self):
        """Initialize FAISS index for vector search"""
        try:
            # This is a stub for FAISS initialization
            # In a real implementation, we would load a pre-built FAISS index
            logger.info("FAISS initialization would happen here")
            self.connected = True  # Set to true for demonstration
        except Exception as e:
            logger.error(f"Error initializing FAISS: {str(e)}")
            self.connected = False
    
    def create_embedding(self, text: str, model_key: str = "default") -> List[float]:
        """Create embedding for the input text using specified model"""
        if not self.embedding_models or model_key not in self.embedding_models:
            logger.warning(f"Embedding model {model_key} not available")
            return [0.0] * EMBED_DIM  # Return zero vector as fallback
        
        try:
            return self.embedding_models[model_key].encode(text).tolist()
        except Exception as e:
            logger.error(f"Error creating embedding with model {model_key}: {str(e)}")
            return [0.0] * EMBED_DIM  # Return zero vector as fallback
    
    async def create_embeddings_parallel(self, text: str) -> Dict[str, List[float]]:
        """Create embeddings using multiple models in parallel"""
        if not self.embedding_models:
            logger.warning("No embedding models available")
            return {"default": [0.0] * EMBED_DIM}
        
        results = {}
        
        # Create tasks for each embedding model
        loop = asyncio.get_running_loop()
        tasks = []
        
        for model_name, model in self.embedding_models.items():
            tasks.append(
                loop.run_in_executor(
                    self.executor,
                    lambda m=model, t=text: m.encode(t).tolist()
                )
            )
        
        # Wait for all embeddings to complete
        embeddings = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, (model_name, _) in enumerate(self.embedding_models.items()):
            if isinstance(embeddings[i], Exception):
                logger.error(f"Error creating embedding with model {model_name}: {str(embeddings[i])}")
                results[model_name] = [0.0] * EMBED_DIM
            else:
                results[model_name] = embeddings[i]
        
        return results
    
    async def enhance_query(self, query: str) -> List[str]:
        """Enhance the query with query expansion techniques"""
        enhanced_queries = [query.strip()]  # Start with the original query
        
        if not USE_QUERY_EXPANSION:
            return enhanced_queries
        
        try:
            # Simple keyword extraction and combination
            keywords = self._extract_keywords(query)
            if keywords:
                # Create variations by combining keywords differently
                enhanced_queries.extend([
                    " ".join(kw for kw in keywords if kw != k)
                    for k in keywords
                ])
            
            # Remove duplicates while preserving order
            seen = set()
            enhanced_queries = [q for q in enhanced_queries if not (q in seen or seen.add(q))]
            
            logger.debug(f"Enhanced query '{query}' into {len(enhanced_queries)} variations")
            return enhanced_queries
            
        except Exception as e:
            logger.error(f"Error enhancing query: {str(e)}")
            return [query.strip()]  # Return original query on error
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from the query"""
        # This is a simple implementation - in production, use a proper NLP library
        # Remove common stop words
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
                     'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'like',
                     'through', 'over', 'before', 'after', 'between', 'under', 'during'}
        
        # Tokenize and filter
        words = query.lower().split()
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    async def retrieve(
        self, 
        query: str, 
        persona: str, 
        top_k: int = NUM_RESULTS
    ) -> Optional[str]:
        """Perform optimized retrieval with caching and parallel processing"""
        if not query.strip():
            return None
            
        start_time = time.time()
        
        # Create cache key
        cache_key = self._create_cache_key(query, persona)
        
        # Check cache first
        cached_result = self.query_cache.get(cache_key)
        if cached_result:
            logger.debug(f"Cache hit for query: {query}")
            retrieval_time = time.time() - start_time
            logger.info(f"Retrieved from cache in {retrieval_time:.2f} seconds")
            return cached_result
        
        # If vector store not available, use fallback
        if not self.connected:
            return await self._fallback_retrieval(query, persona)
        
        try:
            # Use hybrid search if enabled
            if ENABLE_HYBRID_SEARCH:
                results = await self._hybrid_search(query, persona, top_k)
            else:
                results = await self._standard_search(query, persona, top_k)
            
            # Format results if any were found
            if results and len(results) > 0:
                context = self._format_results(results, query, persona)
                
                # Cache the result
                self.query_cache.put(cache_key, context)
                
                # Log retrieval metrics
                retrieval_time = time.time() - start_time
                logger.info(f"Retrieved {len(results)} documents in {retrieval_time:.2f} seconds")
                
                return context
            else:
                logger.warning(f"No results found for query: {query}")
                return None
                
        except Exception as e:
            logger.error(f"Error in RAG retrieval: {str(e)}")
            return await self._fallback_retrieval(query, persona)
    
    async def _hybrid_search(self, query: str, persona: str, top_k: int) -> List[Dict]:
        """Perform hybrid search with multiple embedding models and query enhancement"""
        all_results = []
        seen_ids = set()
        
        # Enhance the query
        enhanced_queries = await self.enhance_query(query)
        
        # Create embeddings for all queries with all models
        for enhanced_query in enhanced_queries:
            # Create embeddings with all models in parallel
            if USE_PARALLEL_EMBEDDING and len(self.embedding_models) > 1:
                embeddings = await self.create_embeddings_parallel(enhanced_query)
            else:
                # Sequential embedding creation
                embeddings = {}
                for model_key in self.embedding_models:
                    embeddings[model_key] = self.create_embedding(enhanced_query, model_key)
            
            # Perform search with each embedding
            for model_key, embedding in embeddings.items():
                batch_results = await self._vector_search(
                    embedding, persona, top_k=top_k
                )
                
                # Add unique results to the combined list
                for result in batch_results:
                    if result['id'] not in seen_ids:
                        all_results.append(result)
                        seen_ids.add(result['id'])
        
        # Rerank combined results if enabled and available
        if ENABLE_RERANKING and self.reranker and len(all_results) > 0:
            all_results = await self._rerank_results(query, all_results)
        
        # Return top K from all combined results
        return all_results[:top_k]
    
    async def _standard_search(self, query: str, persona: str, top_k: int) -> List[Dict]:
        """Perform standard vector search with a single embedding model"""
        # Create embedding for the query
        embedding = self.create_embedding(query)
        
        # Perform search
        results = await self._vector_search(embedding, persona, top_k=top_k)
        
        return results
    
    async def _vector_search(
        self,
        embedding: List[float],
        persona: str,
        top_k: int = NUM_RESULTS
    ) -> List[Dict]:
        """Perform vector search using the configured vector store"""
        if VECTOR_STORE_TYPE == "milvus" and self.vector_store:
            return await self._milvus_search(embedding, persona, top_k)
        elif VECTOR_STORE_TYPE == "faiss" and FAISS_AVAILABLE:
            return await self._faiss_search(embedding, persona, top_k)
        else:
            logger.warning("No vector store available for search")
            return []
    
    async def _milvus_search(
        self,
        embedding: List[float],
        persona: str,
        top_k: int
    ) -> List[Dict]:
        """Perform search in Milvus vector database"""
        try:
            # Search parameters
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            
            # Run search in executor to avoid blocking
            loop = asyncio.get_running_loop()
            
            # Define search function for executor
            def do_search():
                return self.vector_store.search(
                    data=[embedding],
                    anns_field="embedding",
                    param=search_params,
                    limit=top_k * 2,  # Get more results for filtering
                    expr=f"persona == \"{persona}\" || persona == \"general\"",
                    output_fields=["id", "content", "source", "relevance", "persona", "timestamp"]
                )
            
            # Execute search in thread pool
            milvus_results = await loop.run_in_executor(self.executor, do_search)
            
            if not milvus_results or not milvus_results[0]:
                return []
            
            # Convert Milvus results to our format
            results = []
            for hit in milvus_results[0]:
                entity = hit.entity
                result = {
                    'id': entity.get('id', ''),
                    'content': entity.get('content', 'No content available'),
                    'source': entity.get('source', 'Unknown source'),
                    'persona': entity.get('persona', 'general'),
                    'score': float(hit.score),
                    'timestamp': entity.get('timestamp', datetime.now().isoformat())
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Milvus search error: {str(e)}")
            return []
    
    async def _faiss_search(
        self,
        embedding: List[float],
        persona: str,
        top_k: int
    ) -> List[Dict]:
        """Perform search using FAISS index (placeholder)"""
        # This would be implemented with a real FAISS index
        logger.info("FAISS search would be performed here")
        return []
    
    async def _rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """Rerank results using a cross-encoder model"""
        try:
            # Create input pairs for reranker
            pairs = [(query, result['content']) for result in results]
            
            # Run reranking in executor to avoid blocking
            loop = asyncio.get_running_loop()
            
            # Define reranking function for executor
            def do_rerank():
                return self.reranker.predict(pairs)
            
            # Execute reranking in thread pool
            scores = await loop.run_in_executor(self.executor, do_rerank)
            
            # Update results with new scores
            for i, score in enumerate(scores):
                results[i]['score'] = float(score)
            
            # Sort by new scores
            results.sort(key=lambda x: x['score'], reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Error during reranking: {str(e)}")
            return results  # Return original results on error
    
    def _create_cache_key(self, query: str, persona: str) -> str:
        """Create a deterministic cache key for a query + persona combination"""
        key_string = f"{query.lower().strip()}:{persona}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _format_results(self, results: List[Dict], query: str, persona: str) -> str:
        """Format search results into a context string with knowledge fusion"""
        if ENABLE_KNOWLEDGE_FUSION:
            # Knowledge fusion combines and restructures information
            return self._fuse_knowledge(results, query, persona)
        else:
            # Standard formatting just lists results
            return self._format_standard(results, query)
    
    def _format_standard(self, results: List[Dict], query: str) -> str:
        """Format search results into a standard context string"""
        context_parts = []
        
        # Add a header
        context_parts.append(f"FACTUAL CONTEXT FOR: {query}\n")
        
        # Add each result with metadata
        for i, result in enumerate(results):
            content = result.get('content', 'No content available')
            source = result.get('source', 'Unknown source')
            persona = result.get('persona', 'general')
            score = result.get('score', 0.0)
            
            context_parts.append(f"[FACT {i+1}] {content}")
            context_parts.append(f"Source: {source} | Relevance: {score:.2f} | Category: {persona}")
            context_parts.append("")  # Empty line for separation
        
        # Add a footer
        context_parts.append("END OF FACTUAL CONTEXT")
        
        return "\n".join(context_parts)
    
    def _fuse_knowledge(self, results: List[Dict], query: str, persona: str) -> str:
        """Fuse knowledge from multiple sources into a coherent context"""
        # Group facts by topic
        topics = self._cluster_by_topic(results)
        
        context_parts = []
        
        # Add a header
        context_parts.append(f"FACTUAL CONTEXT FOR: {query}\n")
        
        # Process each topic
        for topic_idx, (topic, topic_results) in enumerate(topics.items()):
            # Add topic header if we have multiple topics
            if len(topics) > 1:
                context_parts.append(f"[TOPIC {topic_idx+1}] {topic.upper()}")
            
            # Add synthesized facts for this topic
            facts = self._synthesize_facts(topic_results)
            for i, (fact, sources) in enumerate(facts):
                context_parts.append(f"[FACT {topic_idx+1}.{i+1}] {fact}")
                
                # Add sources
                source_list = ", ".join(sources)
                context_parts.append(f"Sources: {source_list}")
                context_parts.append("")  # Empty line for separation
        
        # Add a footer
        context_parts.append("END OF FACTUAL CONTEXT")
        
        return "\n".join(context_parts)
    
    def _cluster_by_topic(self, results: List[Dict]) -> Dict[str, List[Dict]]:
        """Group results by inferred topic"""
        # This is a simplified implementation
        # A production version would use proper clustering algorithms
        
        # For now, we'll use "persona" as the topic
        topics = {}
        
        for result in results:
            topic = result.get('persona', 'general')
            if topic not in topics:
                topics[topic] = []
            topics[topic].append(result)
        
        return topics
    
    def _synthesize_facts(self, results: List[Dict]) -> List[Tuple[str, List[str]]]:
        """Synthesize consistent facts from multiple results"""
        # For now, we just deduplicate and list sources
        # A production version would use NLP to merge similar facts
        facts = []
        seen_content = set()
        
        for result in results:
            content = result.get('content', '').strip()
            source = result.get('source', 'Unknown')
            
            # Skip if empty or seen before
            if not content or content in seen_content:
                continue
            
            # Check for similar content (very basic check)
            similar_idx = None
            for i, (existing_fact, _) in enumerate(facts):
                # Simple similarity: more than 70% of words match
                if self._text_similarity(content, existing_fact) > 0.7:
                    similar_idx = i
                    break
            
            if similar_idx is not None:
                # Add source to existing fact
                _, sources = facts[similar_idx]
                if source not in sources:
                    sources.append(source)
            else:
                # Add as new fact
                facts.append((content, [source]))
                seen_content.add(content)
        
        return facts
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity based on word overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    async def _fallback_retrieval(self, query: str, persona: str) -> Optional[str]:
        """Fallback retrieval when vector store is not available"""
        logger.info("Using fallback retrieval method")
        
        # Static knowledge base for fallback (would be expanded in production)
        knowledge_base = {
            "economy": {
                "content": "The economy includes topics like inflation, jobs, and fiscal policy.",
                "source": "Economic principles"
            },
            "healthcare": {
                "content": "Healthcare involves policies on insurance, medical costs, and public health.",
                "source": "Healthcare guidelines"
            },
            "immigration": {
                "content": "Immigration policies control borders and determine who can legally enter the country.",
                "source": "Immigration laws"
            },
            "climate": {
                "content": "Climate change and environmental policies address global warming and pollution.",
                "source": "Environmental science"
            },
            "taxes": {
                "content": "Tax policies determine how government revenue is collected from individuals and businesses.",
                "source": "Tax code"
            },
            "education": {
                "content": "Education policies address school funding, curriculum standards, and access to higher education.",
                "source": "Education department"
            }
        }
        
        # Extract keywords from query
        keywords = self._extract_keywords(query)
        
        # Find matching topics
        matches = []
        for keyword in keywords:
            for topic, data in knowledge_base.items():
                if keyword in topic or topic in keyword:
                    matches.append({
                        "id": topic,
                        "content": data["content"],
                        "source": data["source"],
                        "persona": "general",
                        "score": 0.8
                    })
        
        # Format matches if found
        if matches:
            return self._format_standard(matches, query)
        
        return None
    
    def close(self):
        """Close connections and clean up resources"""
        try:
            # Clear cache
            self.query_cache.clear()
            
            # Close vector store connection
            if VECTOR_STORE_TYPE == "milvus" and MILVUS_AVAILABLE:
                connections.disconnect("default")
                logger.info("Disconnected from Milvus")
            
            # Shutdown thread pool
            self.executor.shutdown(wait=False)
            
        except Exception as e:
            logger.error(f"Error closing RAG system: {str(e)}")


# Singleton instance
rag_engine = HybridRAGEngine()

# Async convenience function for retrieval
async def get_context(query: str, persona: str) -> Optional[str]:
    """Get context for a query using the RAG engine"""
    return await rag_engine.retrieve(query, persona) 