"""Cache configuration settings."""

import os
from typing import Dict, Union

from .env_utils import get_env_int

# Cache settings
CACHE_EXPIRY_HOURS = get_env_int("CACHE_EXPIRY_HOURS", 24)
RESPONSE_CACHE_SIZE = get_env_int("RESPONSE_CACHE_SIZE", 1000)

# Cache configuration
CACHE_CONFIG: Dict[str, Union[int, bool]] = {
    "ttl": CACHE_EXPIRY_HOURS * 3600,  # Convert hours to seconds
    "maxsize": RESPONSE_CACHE_SIZE,
    "typed": False
}
