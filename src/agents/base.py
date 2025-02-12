"""Base agent class for PoliticianAI."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from src.utils import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Abstract base class for all agents."""
    
    def __init__(self):
        """Initialize the agent."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def process(
        self,
        input_data: Any,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process input data and return results.
        
        Args:
            input_data: Input data to process
            context: Optional context dictionary
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary containing processing results
        """
        raise NotImplementedError
    
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate input data.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if valid, False otherwise
        """
        return True
    
    def preprocess(
        self,
        input_data: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Preprocess input data before main processing.
        
        Args:
            input_data: Input data to preprocess
            context: Optional context dictionary
            
        Returns:
            Preprocessed input data
        """
        return input_data
    
    def postprocess(
        self,
        output_data: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Postprocess output data after main processing.
        
        Args:
            output_data: Output data to postprocess
            context: Optional context dictionary
            
        Returns:
            Postprocessed output data
        """
        return output_data
    
    def handle_error(self, error: Exception) -> Dict[str, Any]:
        """
        Handle processing errors.
        
        Args:
            error: Exception that occurred
            
        Returns:
            Error response dictionary
        """
        self.logger.error(f"Error in {self.__class__.__name__}: {str(error)}")
        return {
            "success": False,
            "error": str(error),
            "error_type": error.__class__.__name__
        }
    
    def __call__(
        self,
        input_data: Any,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process input data with error handling.
        
        Args:
            input_data: Input data to process
            context: Optional context dictionary
            **kwargs: Additional keyword arguments
            
        Returns:
            Processing results or error response
        """
        try:
            # Validate input
            if not self.validate_input(input_data):
                raise ValueError("Invalid input data")
            
            # Preprocess
            preprocessed_data = self.preprocess(input_data, context)
            
            # Process
            result = self.process(preprocessed_data, context, **kwargs)
            
            # Postprocess
            final_result = self.postprocess(result, context)
            
            return {
                "success": True,
                "result": final_result
            }
            
        except Exception as e:
            return self.handle_error(e)
