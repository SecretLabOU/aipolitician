"""Shared utility functions used in the project."""

import os
import json
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

from langchain_core.language_models import BaseLLM
from langchain_community.llms import LlamaCpp, Ollama
from langchain_openai import OpenAI


def format_conversation_for_prompt(conversation_history: List[Dict[str, str]], max_length: int = 5) -> str:
    """Format conversation history for prompts.
    
    Args:
        conversation_history: A list of message dictionaries
        max_length: Maximum number of messages to include
        
    Returns:
        Formatted conversation history as a string
    """
    # Take only the most recent messages
    recent_history = conversation_history[-max_length:] if len(conversation_history) > max_length else conversation_history
    
    formatted = ""
    for message in recent_history:
        speaker = message.get("speaker", "Unknown")
        text = message.get("text", "")
        formatted += f"{speaker}: {text}\n\n"
    
    return formatted


def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load and parse a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Parsed JSON content as a dictionary
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: File not found: {file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON in file: {file_path}")
        return {}


def get_model_config() -> Dict[str, Any]:
    """Get the model configuration.
    
    Returns:
        Model configuration dictionary
    """
    # Look for config.json in models directory
    config_path = Path(__file__).parent.parent.parent / "models" / "config.json"
    
    if config_path.exists():
        return load_json_file(str(config_path))
    else:
        # Default configuration
        return {
            "models": {
                "mistral": {
                    "type": "ollama",
                    "model": "mistral",
                    "temperature": 0.7,
                    "description": "Mistral 7B instruct model via Ollama"
                }
            },
            "default_model": "mistral"
        }