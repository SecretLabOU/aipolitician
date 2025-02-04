from typing import Any, Dict, List, Optional
from langchain.agents import Tool
from langchain.schema import BaseMemory
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from src.config import EMBEDDING_MODEL, FAISS_INDEX_PATH

class BaseAgent:
    """Base class for all agents in the system"""
    
    def __init__(
        self,
        name: str,
        description: str,
        tools: Optional[List[Tool]] = None,
        memory: Optional[BaseMemory] = None,
        verbose: bool = False
    ):
        self.name = name
        self.description = description
        self.tools = tools or []
        self.memory = memory
        self.verbose = verbose
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cuda'}
        )
        
        # Initialize vector store if path exists
        if FAISS_INDEX_PATH.exists():
            self.vector_store = FAISS.load_local(
                str(FAISS_INDEX_PATH),
                self.embeddings
            )
        else:
            self.vector_store = None

    def add_tool(self, tool: Tool) -> None:
        """Add a tool to the agent"""
        self.tools.append(tool)

    def add_tools(self, tools: List[Tool]) -> None:
        """Add multiple tools to the agent"""
        self.tools.extend(tools)

    def set_memory(self, memory: BaseMemory) -> None:
        """Set the agent's memory"""
        self.memory = memory

    async def arun(self, query: str) -> str:
        """
        Asynchronously run the agent
        
        Args:
            query: The input query to process
            
        Returns:
            str: The agent's response
        """
        raise NotImplementedError("Subclasses must implement arun method")

    def run(self, query: str) -> str:
        """
        Synchronously run the agent
        
        Args:
            query: The input query to process
            
        Returns:
            str: The agent's response
        """
        raise NotImplementedError("Subclasses must implement run method")

    def _validate_input(self, query: str) -> bool:
        """
        Validate the input query
        
        Args:
            query: The input query to validate
            
        Returns:
            bool: Whether the input is valid
        """
        if not query or not isinstance(query, str):
            return False
        return True

    def _format_response(self, response: Any) -> str:
        """
        Format the response for output
        
        Args:
            response: The raw response to format
            
        Returns:
            str: The formatted response
        """
        if isinstance(response, str):
            return response
        elif isinstance(response, dict):
            return str(response.get('output', response))
        return str(response)

    def _get_relevant_context(self, query: str, k: int = 3) -> List[str]:
        """
        Get relevant context from the vector store
        
        Args:
            query: The query to find context for
            k: Number of relevant documents to retrieve
            
        Returns:
            List[str]: List of relevant context strings
        """
        if not self.vector_store:
            return []
            
        docs = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

    def _update_memory(self, query: str, response: str) -> None:
        """
        Update the agent's memory with the interaction
        
        Args:
            query: The input query
            response: The agent's response
        """
        if self.memory:
            self.memory.save_context(
                {"input": query},
                {"output": response}
            )
