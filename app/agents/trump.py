from typing import Dict, Optional
from .base import PoliticalAgent
from langchain_core.language_models import BaseLLM
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

class TrumpAgent(PoliticalAgent):
    def __init__(self):
        personality_traits = {
            "assertiveness": 0.9,
            "directness": 0.95,
            "confidence": 0.95,
            "informality": 0.8,
            "repetition": 0.7
        }
        super().__init__("Donald Trump", personality_traits)
        
        # Initialize model and tokenizer
        self.model_name = "mistralai/Mistral-7B-v0.1"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=True
        )
        
        # Set up generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto"
        )
        
        # Trump-specific prompt template
        self.prompt_template = """You are Donald Trump, the 45th President of the United States. 
        Respond to the following message in your characteristic style, incorporating your unique speech patterns, 
        policy positions, and personality. Keep responses concise and impactful.

        Previous conversation:
        {chat_history}

        User: {message}
        Trump:"""

    def _format_response(self, response: str) -> str:
        """Format response in Trump's characteristic style."""
        # Add typical Trump phrases and formatting
        response = response.replace(".", "!")
        
        # Add emphasis to key words
        for word in ["great", "huge", "tremendous", "amazing"]:
            response = response.replace(word, word.upper())
        
        return response

    async def generate_response(
        self,
        message: str,
        session_id: Optional[str] = None
    ) -> Dict:
        # Get memory and format chat history
        memory, session_id = self._get_memory(session_id)
        chat_history = "\n".join([
            f"{'User' if isinstance(msg, HumanMessage) else 'Trump'}: {msg.content}"
            for msg in memory.chat_memory.messages
        ])
        
        # Create prompt
        prompt = self.prompt_template.format(
            chat_history=chat_history,
            message=message
        )
        
        # Generate response
        generated = self.generator(
            prompt,
            max_length=200,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )[0]["generated_text"]
        
        # Extract the response part
        response = generated.split("Trump:")[-1].strip()
        
        # Format response and update memory
        formatted_response = self._format_response(response)
        memory.chat_memory.add_message(HumanMessage(content=message))
        memory.chat_memory.add_message(AIMessage(content=formatted_response))
        
        # Analyze sentiment
        sentiment = self._analyze_sentiment(message)
        
        return {
            "response": formatted_response,
            "session_id": session_id,
            "sentiment": sentiment,
            "context": {
                "agent_name": self.name,
                "personality_traits": str(self.personality_traits)
            }
        }

    def set_gpu_device(self, device_id: int):
        """Set GPU device for all models."""
        super().set_gpu_device(device_id)
        if torch.cuda.is_available():
            self.model = self.model.to(f"cuda:{device_id}")
