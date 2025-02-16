from typing import Dict, Optional
from .base import PoliticalAgent, AgentState
from ..utils.model_manager import ModelManager
from ..utils.exceptions import ModelGenerationError, ModelLoadError

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
        
        try:
            # Initialize model through ModelManager
            model_dict = ModelManager.get_model("mistralai/Mistral-7B-v0.1")
            self.generator = model_dict["generator"]
        except Exception as e:
            raise ModelLoadError(f"Failed to load Trump's model: {str(e)}")

    def _generate_response_node(self, state: AgentState) -> AgentState:
        """Generate response using Trump's characteristic style."""
        try:
            # Format conversation history
            chat_history = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in state["messages"]
            ])
            
            # Create prompt
            prompt = f"""You are Donald Trump, the 45th President of the United States. 
            Respond to the following message in your characteristic style, incorporating your unique speech patterns, 
            policy positions, and personality. Keep responses concise and impactful.

            Previous conversation:
            {chat_history}

            User: {state['messages'][-1]['content']}
            Trump:"""
            
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
            response = generated.split("Trump:")[-1].strip()
            state["messages"].append({
                "role": "assistant",
                "content": response
            })
            
            # Set next step
            state["next_step"] = "format_response"
            return state
            
        except Exception as e:
            raise ModelGenerationError(f"Failed to generate Trump's response: {str(e)}")

    def _format_response_node(self, state: AgentState) -> AgentState:
        """Format the response in Trump's characteristic style."""
        try:
            response = state["messages"][-1]["content"]
            
            # Add typical Trump formatting
            response = response.replace(".", "!")
            
            # Add emphasis to key words
            trump_keywords = [
                "great", "huge", "tremendous", "amazing",
                "winning", "best", "beautiful", "perfect"
            ]
            
            for word in trump_keywords:
                if word.lower() in response.lower():
                    response = response.replace(word.lower(), word.upper())
                    response = response.replace(word.capitalize(), word.UPPER())
            
            # Update the formatted response
            state["messages"][-1]["content"] = response
            
            # Set next step to end
            state["next_step"] = "end"
            return state
            
        except Exception as e:
            raise ModelGenerationError(f"Failed to format Trump's response: {str(e)}")

    def set_gpu_device(self, device_id: int):
        """Set GPU device for all models."""
        super().set_gpu_device(device_id)
        try:
            ModelManager.set_gpu_device("mistralai/Mistral-7B-v0.1", device_id)
        except Exception as e:
            print(f"Warning: Failed to set GPU device for Trump's model: {str(e)}")
