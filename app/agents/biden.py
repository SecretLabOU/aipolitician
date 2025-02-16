from typing import Dict, Optional
from .base import PoliticalAgent, AgentState
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import random

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
        
        # Biden's characteristic phrases
        self.biden_phrases = [
            "Look, folks",
            "Here's the deal",
            "I'm serious",
            "Not a joke"
        ]

    def _generate_response_node(self, state: AgentState) -> AgentState:
        """Generate response using Biden's characteristic style."""
        # Format conversation history
        chat_history = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in state["messages"]
        ])
        
        # Create prompt
        prompt = f"""You are Joe Biden, the 46th President of the United States. 
        Respond to the following message in your characteristic style, incorporating your unique speech patterns, 
        policy positions, and personality. Focus on unity, empathy, and experience while maintaining your 
        distinctive communication style.

        Previous conversation:
        {chat_history}

        User: {state['messages'][-1]['content']}
        Biden:"""
        
        # Generate response
        generated = self.generator(
            prompt,
            max_length=200,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )[0]["generated_text"]
        
        # Extract and add response to state
        response = generated.split("Biden:")[-1].strip()
        state["messages"].append({
            "role": "assistant",
            "content": response
        })
        
        return state

    def _format_response_node(self, state: AgentState) -> AgentState:
        """Format the response in Biden's characteristic style."""
        response = state["messages"][-1]["content"]
        
        # Add Biden's characteristic phrases if not present
        if not any(phrase in response for phrase in self.biden_phrases):
            phrase = random.choice(self.biden_phrases)
            response = f"{phrase}, {response}"
        
        # Add emphasis to key policy words
        policy_words = ["democracy", "unity", "middle class", "america"]
        for word in policy_words:
            if word.lower() in response.lower():
                response = response.replace(word, word.upper())
        
        # Update the formatted response
        state["messages"][-1]["content"] = response
        return state

    def set_gpu_device(self, device_id: int):
        """Set GPU device for all models."""
        super().set_gpu_device(device_id)
        if torch.cuda.is_available():
            self.model = self.model.to(f"cuda:{device_id}")
