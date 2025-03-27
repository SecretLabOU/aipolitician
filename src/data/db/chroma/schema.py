import os
import sys
import logging
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import Optional, Dict, Any, List
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DB_PATH = os.path.expanduser("~/political_db")
DEFAULT_COLLECTION_NAME = "political_figures"

def setup_permissions(db_path: str) -> bool:
    """
    Set up proper permissions for the database directory to ensure
    it's readable by all users but only writable by the owner.
    
    Args:
        db_path: Path to the database directory
        
    Returns:
        bool: True if permissions were set successfully
    """
    try:
        # Create directory if it doesn't exist
        if not os.path.exists(db_path):
            os.makedirs(db_path, exist_ok=True)
            logger.info(f"Created database directory at {db_path}")
            
        # Set permissions: 755 (rwxr-xr-x)
        # Owner can read, write, execute
        # Others can read and execute, but not write
        os.chmod(db_path, 0o755)
        logger.info(f"Set permissions on {db_path} to 755 (rwxr-xr-x)")
        
        return True
    except Exception as e:
        logger.error(f"Failed to set permissions on {db_path}: {e}")
        return False

def get_embedding_function():
    """
    Get the embedding function for the BGE-Small-EN model using HuggingFace Transformers directly.
    
    Returns:
        embedding_function: The embedding function to use
    """
    try:
        # Direct implementation using HuggingFace Transformers
        from transformers import AutoTokenizer, AutoModel
        import torch
        
        class BGEEmbeddingFunction:
            def __init__(self, model_name="BAAI/bge-small-en"):
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model.to(self.device)
                logger.info(f"Loaded BGE-Small-EN model on {self.device}")
            
            def __call__(self, input):
                """
                Generate embeddings for the input texts.
                
                Args:
                    input: List of texts to generate embeddings for
                    
                Returns:
                    List of embeddings
                """
                if not input:
                    return []
                
                # Tokenize sentences
                encoded_input = self.tokenizer(
                    input, 
                    padding=True, 
                    truncation=True, 
                    max_length=512, 
                    return_tensors='pt'
                ).to(self.device)
                
                # Compute token embeddings
                with torch.no_grad():
                    model_output = self.model(**encoded_input)
                    # Use CLS token embedding
                    embeddings = model_output.last_hidden_state[:, 0]
                    
                    # Normalize embeddings
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    
                # Convert from PyTorch tensors to numpy arrays
                embeddings = embeddings.cpu().numpy()
                
                return embeddings.tolist()
            
        embedding_func = BGEEmbeddingFunction()
        logger.info("Initialized BGE-Small-EN embedding function using HuggingFace Transformers")
        return embedding_func
        
    except Exception as e:
        logger.error(f"Failed to initialize embedding function: {e}")
        # Return None to indicate failure
        return None

def connect_to_chroma(db_path: str = DEFAULT_DB_PATH, persist: bool = True) -> Optional[chromadb.Client]:
    """
    Connect to a ChromaDB instance.
    
    Args:
        db_path: Path to the ChromaDB database
        persist: Whether to persist the database to disk
        
    Returns:
        chromadb.Client or None: ChromaDB client if successful, None otherwise
    """
    try:
        # Ensure directory exists and has proper permissions
        if not setup_permissions(db_path):
            logger.error("Failed to set up database directory permissions")
            return None
            
        # Create the client
        if persist:
            client = chromadb.PersistentClient(
                path=db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info(f"Connected to persistent ChromaDB at {db_path}")
        else:
            client = chromadb.EphemeralClient(
                settings=Settings(
                    anonymized_telemetry=False
                )
            )
            logger.info("Connected to ephemeral ChromaDB (in-memory)")
            
        return client
    except Exception as e:
        logger.error(f"Failed to connect to ChromaDB: {e}")
        return None

def get_collection(client: chromadb.Client, collection_name: str = DEFAULT_COLLECTION_NAME, 
                   create_if_not_exists: bool = True) -> Optional[chromadb.Collection]:
    """
    Get a collection from ChromaDB.
    
    Args:
        client: ChromaDB client
        collection_name: Name of the collection
        create_if_not_exists: Whether to create the collection if it doesn't exist
        
    Returns:
        chromadb.Collection or None: Collection if successful, None otherwise
    """
    try:
        # Get embedding function
        embedding_func = get_embedding_function()
        if not embedding_func:
            raise ValueError("Could not initialize embedding function")
            
        # Check if collection exists (compatible with ChromaDB v0.6.0)
        try:
            collections = client.list_collections()
            # In ChromaDB v0.6.0+, list_collections returns just the names
            collection_exists = collection_name in collections
            logger.info(f"Found collections: {collections}")
        except Exception as e:
            # Fallback for older versions or if there's an error
            logger.warning(f"Error checking collections with list_collections: {e}")
            try:
                # Try to get the collection directly to check if it exists
                client.get_collection(name=collection_name)
                collection_exists = True
            except Exception:
                collection_exists = False
        
        if collection_exists:
            logger.info(f"Collection '{collection_name}' already exists")
            try:
                collection = client.get_collection(
                    name=collection_name,
                    embedding_function=embedding_func
                )
                return collection
            except Exception as e:
                logger.error(f"Error getting existing collection: {e}")
                if not create_if_not_exists:
                    return None
                    
                # Try to delete and recreate if we can't get it properly
                try:
                    client.delete_collection(name=collection_name)
                    logger.info(f"Deleted problematic collection '{collection_name}'")
                    collection_exists = False
                except Exception as delete_error:
                    logger.error(f"Error deleting problematic collection: {delete_error}")
                    return None
        
        if not collection_exists and create_if_not_exists:
            logger.info(f"Creating collection '{collection_name}'")
            collection = client.create_collection(
                name=collection_name,
                embedding_function=embedding_func,
                metadata={"description": "Political figures database for AI debate system"}
            )
            return collection
        else:
            logger.warning(f"Collection '{collection_name}' does not exist and create_if_not_exists is False")
            return None
    except Exception as e:
        logger.error(f"Failed to get collection: {e}")
        return None

def delete_collection(client: chromadb.Client, collection_name: str = DEFAULT_COLLECTION_NAME) -> bool:
    """
    Delete a collection from ChromaDB.
    
    Args:
        client: ChromaDB client
        collection_name: Name of the collection to delete
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        client.delete_collection(name=collection_name)
        logger.info(f"Deleted collection '{collection_name}'")
        return True
    except Exception as e:
        logger.error(f"Failed to delete collection '{collection_name}': {e}")
        return False
        
def initialize_database(db_path: str = DEFAULT_DB_PATH, 
                        collection_name: str = DEFAULT_COLLECTION_NAME) -> Dict[str, Any]:
    """
    Initialize the ChromaDB database by connecting and creating a collection.
    
    Args:
        db_path: Path to the ChromaDB database
        collection_name: Name of the collection
        
    Returns:
        Dict with 'client' and 'collection' keys
    """
    # Connect to ChromaDB
    client = connect_to_chroma(db_path=db_path)
    if not client:
        raise ConnectionError("Could not connect to ChromaDB")
    
    # Get or create collection
    collection = get_collection(client, collection_name=collection_name)
    if not collection:
        raise ValueError(f"Could not create or get collection '{collection_name}'")
    
    return {
        "client": client,
        "collection": collection
    }

if __name__ == "__main__":
    print("Initializing ChromaDB database schema...")
    db = initialize_database()
    print(f"Schema initialization complete! Using collection: {db['collection'].name}") 