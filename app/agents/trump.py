from typing import Dict, Optional
from .base import PoliticalAgent, AgentState
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
                max_length=200,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
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
            print(f"Error in generate_response_node: {str(e)}")
            raise

    def _format_response_node(self, state: AgentState) -> AgentState:
        """Format the response in Trump's characteristic style."""
        try:
            response = state["messages"][-1]["content"]
            
            # Add typical Trump formatting
            response = response.replace(".", "!")
            
            # Add emphasis to key words
            for word in ["great", "huge", "tremendous", "amazing"]:
                if word.lower() in response.lower():
                    response = response.replace(word.lower(), word.upper())
                    response = response.replace(word.capitalize(), word.upper())
            
            # Update the formatted response
            state["messages"][-1]["content"] = response
            
            # Set next step to end
            state["next_step"] = "end"
            return state
            
        except Exception as e:
            print(f"Error in format_response_node: {str(e)}")
            raise

    def set_gpu_device(self, device_id: int):
        """Set GPU device for all models."""
        super().set_gpu_device(device_id)
        if torch.cuda.is_available():
            self.model = self.model.to(f"cuda:{device_id}")
