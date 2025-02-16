class AIPoliticianError(Exception):
    """Base exception class for AI Politician errors."""
    pass

class ModelError(AIPoliticianError):
    """Raised when there's an error with model operations."""
    pass

class ModelLoadError(ModelError):
    """Raised when there's an error loading a model."""
    pass

class ModelGenerationError(ModelError):
    """Raised when there's an error generating a response."""
    pass

class SessionError(AIPoliticianError):
    """Raised when there's an error with session operations."""
    pass

class SessionNotFoundError(SessionError):
    """Raised when a requested session is not found."""
    pass

class SessionExpiredError(SessionError):
    """Raised when attempting to access an expired session."""
    pass

class AgentError(AIPoliticianError):
    """Raised when there's an error with agent operations."""
    pass

class AgentNotFoundError(AgentError):
    """Raised when a requested agent is not found."""
    pass

class InvalidRequestError(AIPoliticianError):
    """Raised when the request is invalid."""
    pass

class RateLimitError(AIPoliticianError):
    """Raised when rate limit is exceeded."""
    pass

def handle_chat_error(error: Exception) -> dict:
    """Convert exceptions to appropriate HTTP responses."""
    if isinstance(error, ModelLoadError):
        return {
            "status_code": 503,
            "detail": f"Service temporarily unavailable: {str(error)}"
        }
    elif isinstance(error, ModelGenerationError):
        return {
            "status_code": 500,
            "detail": f"Failed to generate response: {str(error)}"
        }
    elif isinstance(error, SessionNotFoundError):
        return {
            "status_code": 404,
            "detail": f"Session not found: {str(error)}"
        }
    elif isinstance(error, SessionExpiredError):
        return {
            "status_code": 410,
            "detail": f"Session expired: {str(error)}"
        }
    elif isinstance(error, AgentNotFoundError):
        return {
            "status_code": 404,
            "detail": f"Agent not found: {str(error)}"
        }
    elif isinstance(error, InvalidRequestError):
        return {
            "status_code": 400,
            "detail": f"Invalid request: {str(error)}"
        }
    elif isinstance(error, RateLimitError):
        return {
            "status_code": 429,
            "detail": "Too many requests. Please try again later."
        }
    else:
        return {
            "status_code": 500,
            "detail": f"Internal server error: {str(error)}"
        }
