from typing import Dict, Optional
from .base import PoliticalAgent
from langchain_core.language_models import BaseLLM
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

class BidenAgent(PoliticalAgent):
    def __init__(self):
        personality_traits = {
            "empathy": 0.9,
            "experience": 0.95,
            "composure": 0.8,
            "formality": 0.7,
            "storytelling": 0.85
        }
        super().__init__("Joe Biden", personality_traits)
        
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
        
        # Biden-specific prompt template
        self.prompt_template = """You are Joe Biden, the 46th President of the United States. 
        Respond to the following message in your characteristic style, incorporating your unique speech patterns, 
        policy positions, and personality. Focus on unity, empathy, and experience while maintaining your 
        distinctive communication style.

        Previous conversation:
        {chat_history}

        User: {message}
        Biden:"""

    def _format_response(self, response: str) -> str:
        """Format response in Biden's characteristic style."""
        # Add typical Biden phrases
        biden_phrases = [
            "Look, folks",
            "Here's the deal",
            "I'm serious",
            "Not a joke"
        ]
        
        # Randomly insert Biden's characteristic phrases
        if not any(phrase in response for phrase in biden_phrases):
            response = f"{biden_phrases[0]}, {response}"
        
        # Add emphasis to key policy words
        policy_words = ["democracy", "unity", "middle class", "america"]
        for word in policy_words:
            if word.lower() in response.lower():
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
            f"{'User' if isinstance(msg, HumanMessage) else 'Biden'}: {msg.content}"
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
        response = generated.split("Biden:")[-1].strip()
        
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
