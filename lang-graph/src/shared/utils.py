"""Shared utility functions used in the project.

Functions:
    format_docs: Convert documents to an xml-formatted string.
    load_chat_model: Load a chat model (local or API-based).
"""

import os
from typing import Optional, Dict, Any

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_community.llms import LlamaCpp, Ollama
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.chat_models import BaseChatModel


def _format_doc(doc: Document) -> str:
    """Format a single document as XML."""
    metadata = doc.metadata or {}
    meta = "".join(f" {k}={v!r}" for k, v in metadata.items())
    if meta:
        meta = f" {meta}"

    return f"<document{meta}>\n{doc.page_content}\n</document>"


def format_docs(docs: Optional[list[Document]]) -> str:
    """Format a list of documents as XML."""
    if not docs:
        return "<documents></documents>"
    formatted = "\n".join(_format_doc(doc) for doc in docs)
    return f"""<documents>
{formatted}
</documents>"""


# Import local model configuration
try:
    from political_agent_graph.local_models import (
        load_model_config,
        get_default_model,
        get_model_details
    )
    # Load the model configuration
    _config = load_model_config()
    _default_model = get_default_model()
except ImportError:
    # Fallback default configuration
    _config = {
        "models": {
            "mixtral": {
                "type": "ollama",
                "model": "mixtral",
                "temperature": 0.7
            }
        }
    }
    _default_model = "mixtral"

# Model provider mappings
MODEL_PROVIDER_MAP = {
    "anthropic": ChatAnthropic,
    "openai": ChatOpenAI,
    "local": None,  # Handled specially
}

def load_chat_model(model_identifier: str, **kwargs) -> BaseChatModel:
    """Load a chat model based on the identifier.
    
    Args:
        model_identifier: String in format 'provider/model' or 'local/model_name'
        
    Returns:
        An instance of a chat model
    """
    # Check for local model specification
    if model_identifier.startswith("local/"):
        _, model_name = model_identifier.split("/", 1)
        return load_local_model(model_name, **kwargs)
    
    # For API-based models
    if "/" in model_identifier:
        provider, model = model_identifier.split("/", 1)
        if provider in MODEL_PROVIDER_MAP:
            model_class = MODEL_PROVIDER_MAP[provider]
            if model_class:
                try:
                    return model_class(model=model, **kwargs)
                except Exception as e:
                    print(f"Warning: Failed to load {model_identifier} with error: {str(e)}")
    
    # Try langchain's init_chat_model
    try:
        if "/" in model_identifier:
            provider, model = model_identifier.split("/", 1)
        else:
            provider = ""
            model = model_identifier
        return init_chat_model(model, model_provider=provider, **kwargs)
    except Exception as e:
        # Fallback to default local model if API access fails
        print(f"Warning: Failed to load {model_identifier}: {e}")
        print(f"Falling back to default local model: {_default_model}")
        return load_local_model(_default_model, **kwargs)

def load_local_model(model_name: str, **kwargs) -> BaseChatModel:
    """Load a local language model.
    
    Args:
        model_name: Name of the local model to load
        
    Returns:
        A chat model instance
    """
    try:
        # Get model details from configuration
        if model_name not in _config["models"]:
            available_models = list(_config["models"].keys())
            print(f"Unknown model: {model_name}. Available models: {available_models}")
            model_name = _default_model
        
        model_config = get_model_details(model_name)
        config = model_config.copy()
        config.update(kwargs)
        
        model_type = config.pop("type")
        
        if model_type == "llamacpp":
            path = config.pop("path")
            # Handle relative paths
            if not os.path.isabs(path):
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                path = os.path.join(base_dir, path)
            
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")
            
            return LlamaCpp(model_path=path, **config)
        
        elif model_type == "ollama":
            ollama_model = config.pop("model")
            return Ollama(model=ollama_model, **config)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        # Last resort fallback to Ollama with mixtral
        print("Using last resort fallback to Ollama with mixtral")
        return Ollama(model="mixtral", temperature=0.7)
