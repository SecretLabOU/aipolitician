"""Environment variable utilities."""

import os
from typing import Any, Optional, Union

def get_env_bool(key: str, default: bool = False) -> bool:
    """
    Get boolean environment variable.
    
    Args:
        key: Environment variable key
        default: Default value if not set
        
    Returns:
        Boolean value
    """
    value = os.getenv(key)
    if value is None:
        return default
    return value.lower() in ("true", "1", "t", "yes", "y")

def get_env_int(key: str, default: int = 0) -> int:
    """
    Get integer environment variable.
    
    Args:
        key: Environment variable key
        default: Default value if not set
        
    Returns:
        Integer value
    """
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default

def get_env_float(key: str, default: float = 0.0) -> float:
    """
    Get float environment variable.
    
    Args:
        key: Environment variable key
        default: Default value if not set
        
    Returns:
        Float value
    """
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default

def get_env_list(
    key: str,
    default: Optional[list] = None,
    separator: str = ","
) -> list:
    """
    Get list environment variable.
    
    Args:
        key: Environment variable key
        default: Default value if not set
        separator: List item separator
        
    Returns:
        List value
    """
    value = os.getenv(key)
    if value is None:
        return default or []
    return [item.strip() for item in value.split(separator) if item.strip()]

def get_env_dict(
    key: str,
    default: Optional[dict] = None,
    separator: str = ",",
    item_separator: str = "="
) -> dict:
    """
    Get dictionary environment variable.
    
    Args:
        key: Environment variable key
        default: Default value if not set
        separator: Dictionary item separator
        item_separator: Key-value separator
        
    Returns:
        Dictionary value
    """
    value = os.getenv(key)
    if value is None:
        return default or {}
    
    result = {}
    for item in value.split(separator):
        if not item.strip():
            continue
        try:
            k, v = item.split(item_separator, 1)
            result[k.strip()] = v.strip()
        except ValueError:
            continue
    return result
