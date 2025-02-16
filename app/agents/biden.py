from typing import Dict, Optional
from .base import PoliticalAgent, AgentState
from ..utils.model_manager import ModelManager
from ..utils.exceptions import ModelGenerationError, ModelLoadError
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
        
        try:
            # Initialize model through ModelManager
            model_dict = ModelManager.get_model("mistralai/Mistral-7B-v0.1")
            self.generator = model_dict["generator"]
        except Exception as e:
            raise ModelLoadError(f"Failed to load Biden's model: {str(e)}")
        
        # Biden's characteristic phrases
        self.biden_phrases = [
            "Look, folks",
            "Here's the deal",
            "I'm serious",
            "Not a joke",
            "Let me be clear",
            "Come on, man",
            "The fact is",
            "And I mean this sincerely"
        ]

    def _generate_response_node(self, state: AgentState) -> AgentState:
        """Generate response using Biden's characteristic style."""
        try:
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
                max_new_tokens=200,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1
            )[0]["generated_text"]
            
            # Extract and add response to state
            response = generated.split("Biden:")[-1].strip()
            state["messages"].append({
                "role": "assistant",
                "content": response
            })
            
            # Set next step
            state["next_step"] = "format_response"
            return state
            
        except Exception as e:
            raise ModelGenerationError(f"Failed to generate Biden's response: {str(e)}")

    def _format_response_node(self, state: AgentState) -> AgentState:
        """Format the response in Biden's characteristic style."""
        try:
            response = state["messages"][-1]["content"]
            
            # Add Biden's characteristic phrases if not present
            if not any(phrase in response for phrase in self.biden_phrases):
                phrase = random.choice(self.biden_phrases)
                response = f"{phrase}, {response}"
            
            # Add emphasis to key policy words
            biden_keywords = [
                "democracy", "unity", "middle class", "america",
                "folks", "truth", "dignity", "respect",
                "working families", "soul of the nation"
            ]
            
            for word in biden_keywords:
                if word.lower() in response.lower():
                    response = response.replace(word.lower(), word.upper())
                    response = response.replace(word.capitalize(), word.UPPER())
            
            # Update the formatted response
            state["messages"][-1]["content"] = response
            
            # Set next step to end
            state["next_step"] = "end"
            return state
            
        except Exception as e:
            raise ModelGenerationError(f"Failed to format Biden's response: {str(e)}")

    def set_gpu_device(self, device_id: int):
        """Set GPU device for all models."""
        super().set_gpu_device(device_id)
        try:
            ModelManager.set_gpu_device("mistralai/Mistral-7B-v0.1", device_id)
        except Exception as e:
            print(f"Warning: Failed to set GPU device for Biden's model: {str(e)}")
