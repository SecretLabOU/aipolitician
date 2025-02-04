from typing import Dict, Optional
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
from sqlalchemy.orm import Session

from .sentiment_agent import SentimentAgent
from .context_agent import ContextAgent
from .response_agent import ResponseAgent
from src.config import MODEL_DIR

class WorkflowManager:
    """Orchestrates the interaction between different agents in the system"""
    
    def __init__(
        self,
        db_session: Session,
        memory: Optional[ConversationBufferMemory] = None,
        verbose: bool = False
    ):
        self.verbose = verbose
        self.memory = memory or ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize agents
        self.sentiment_agent = SentimentAgent(
            memory=self.memory,
            verbose=verbose
        )
        
        self.context_agent = ContextAgent(
            memory=self.memory,
            verbose=verbose
        )
        
        self.response_agent = ResponseAgent(
            db_session=db_session,
            memory=self.memory,
            verbose=verbose
        )
        
        # Initialize LLM for agent coordination
        self.llm = LlamaCpp(
            model_path=str(MODEL_DIR / "llama-2-7b-chat.gguf"),
            temperature=0.1,  # Lower temperature for more focused coordination
            max_tokens=512,
            top_p=0.95,
            n_ctx=2048,
            verbose=verbose
        )
        
        # Set up tools
        self.tools = self._setup_tools()
        
        # Create agent executor
        self.agent_executor = self._create_agent_executor()

    def _setup_tools(self) -> list:
        """Set up tools for the agent"""
        return [
            Tool(
                name="SentimentAnalysis",
                func=self.sentiment_agent.run,
                description="Analyze sentiment and emotional tone of text"
            ),
            Tool(
                name="ContextExtraction",
                func=self.context_agent.run,
                description="Extract context and identify topics"
            ),
            Tool(
                name="ResponseGeneration",
                func=lambda x: self.response_agent.run(
                    x,
                    self.context_agent.run(x),
                    self.sentiment_agent.run(x)
                ),
                description="Generate contextual response"
            )
        ]

    def _create_agent_executor(self) -> AgentExecutor:
        """Create the agent executor with custom prompt"""
        prompt_template = """
        Process the user's input through the following steps:
        1. Analyze sentiment to understand emotional context
        2. Extract topics and context
        3. Generate appropriate response
        
        Previous conversation context:
        {chat_history}
        
        User input: {input}
        
        Think through the steps:
        1) First, determine if we need sentiment analysis
        2) Then, consider what context we need
        3) Finally, decide how to generate the response
        
        Available tools:
        {tools}
        
        Action: """
        
        # Create prompt template
        prompt = StringPromptTemplate(
            template=prompt_template,
            input_variables=["input", "chat_history", "tools"]
        )
        
        # Create LLM chain for the agent
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        
        # Create the agent
        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=None,  # Using default output parser
            stop=["\nObservation:"],
            allowed_tools=[tool.name for tool in self.tools]
        )
        
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=self.verbose
        )

    async def aprocess_input(self, user_input: str) -> Dict:
        """
        Asynchronously process user input through the agent workflow
        
        Args:
            user_input: The user's input text
            
        Returns:
            Dict containing the processed response and metadata
        """
        return await self.agent_executor.arun(input=user_input)

    def process_input(self, user_input: str) -> Dict:
        """
        Process user input through the agent workflow
        
        Args:
            user_input: The user's input text
            
        Returns:
            Dict containing the processed response and metadata
        """
        try:
            # Get sentiment analysis
            sentiment = self.sentiment_agent.run(user_input)
            
            # Get context analysis
            context = self.context_agent.run(user_input)
            
            # Generate response
            response = self.response_agent.run(
                user_input,
                context,
                sentiment
            )
            
            return {
                "response": response,
                "metadata": {
                    "sentiment": sentiment,
                    "context": context
                }
            }
            
        except Exception as e:
            if self.verbose:
                print(f"Error in workflow: {str(e)}")
            return {
                "response": "I encountered an error processing your request.",
                "metadata": {
                    "error": str(e)
                }
            }

    def get_conversation_history(self) -> list:
        """Get the conversation history from memory"""
        return self.memory.chat_memory.messages if self.memory else []

    def clear_memory(self) -> None:
        """Clear the conversation memory"""
        if self.memory:
            self.memory.clear()
