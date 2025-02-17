from abc import ABC, abstractmethod
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

class BaseAgent(ABC):
    def __init__(self):
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=0 if torch.cuda.is_available() else -1
        )
        self.model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
        self.personality_traits = ""
    
    def analyze_sentiment(self, text: str):
        return self.sentiment_analyzer(text)[0]
    
    @abstractmethod
    def format_prompt(self, user_input: str, history: list) -> str:
        pass
    
    def generate_response(self, user_input: str, history: list) -> str:
        sentiment = self.analyze_sentiment(user_input)
        prompt = self.format_prompt(user_input, history)
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        # Create attention mask
        attention_mask = torch.ones_like(inputs.input_ids)
        
        outputs = self.model.generate(
            inputs.input_ids,
            attention_mask=attention_mask,
            max_length=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=3
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
