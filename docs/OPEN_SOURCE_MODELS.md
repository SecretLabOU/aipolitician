# Open Source Models for AI Politician

This document outlines the open-source models used in the AI Politician LangGraph system, their capabilities, and hardware requirements.

## Models

### 1. Context Extraction Model: Mixtral 8x7B Instruct

**Model ID**: `mistralai/Mixtral-8x7B-Instruct-v0.1`

**Purpose**: This model analyzes user input to extract key topics, policy areas, and factual questions. It's used by the Context Agent to enhance the RAG retrieval process.

**Capabilities**:
- High quality text understanding and generation
- Mixture of Experts (MoE) architecture, providing strong performance
- Instruction-tuned for precise following of prompts
- Competitive with GPT-3.5 in many tasks

**Hardware Requirements**:
- VRAM: 16GB+ (with 4-bit quantization)
- GPU: NVIDIA RTX 4080 or better (which you have)

**Fallback Model**: For systems with limited VRAM, the system will automatically switch to TinyLlama (1.1B parameters), which requires only 2-4GB VRAM.

### 2. Sentiment Analysis Model: RoBERTa for Emotion Detection

**Model ID**: `SamLowe/roberta-base-go_emotions`

**Purpose**: This model analyzes the sentiment and emotional content of user messages. It detects multiple emotions and helps determine if deflection is needed.

**Capabilities**:
- Detects 28 different emotions with good accuracy
- Fine-tuned specifically for social media-style text
- Much more nuanced than simple positive/negative sentiment analysis
- Lightweight and fast inference

**Hardware Requirements**:
- VRAM: 2GB
- Works well on any GPU, including your RTX 4060 Ti

### 3. Response Generation Models: Fine-tuned Mistral 7B

**Model IDs**:
- `nnat03/biden-mistral-adapter` (your Biden model)
- `nnat03/trump-mistral-adapter` (your Trump model)

**Purpose**: These are your existing fine-tuned models that generate politician-specific responses based on context and sentiment analysis.

**Capabilities**:
- Character-specific response generation
- Maintenance of consistent political persona
- Instruction-following with your custom fine-tuning

**Hardware Requirements**:
- VRAM: 8GB+ (with 4-bit quantization)
- GPU: NVIDIA RTX 4060 Ti or better (which you have)

## Hardware Utilization

Your system has:
- NVIDIA GeForce RTX 4080 (16GB VRAM)
- NVIDIA GeForce RTX 4060 Ti (16GB VRAM)
- Quadro RTX 8000 (48GB VRAM)

With this hardware, the system will:
1. Load the context model (Mixtral 8x7B) on the RTX 8000
2. Load the sentiment model (RoBERTa) on the RTX 4060 Ti
3. Load the politician models (fine-tuned Mistral 7B) on the RTX 4080

This configuration provides optimal performance without requiring any OpenAI API access.

## Performance Considerations

- **4-bit Quantization**: All large language models use 4-bit quantization to reduce VRAM usage
- **Model Caching**: Models remain loaded in memory between requests for better performance
- **Graceful Degradation**: If a model fails to load, simpler fallback methods are used

## Alternative Models

If you prefer smaller models or different capabilities, you can modify `config.py` to use these alternatives:

### For Context Extraction:
- `MBZUAI/LaMini-Flan-T5-783M` (lightweight, 2GB VRAM)
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (balanced, 4GB VRAM)
- `mosaicml/mpt-7b-instruct` (powerful alternative, 14GB VRAM)

### For Sentiment Analysis:
- `cardiffnlp/twitter-roberta-base-sentiment-latest` (fast, simple sentiment)
- `finiteautomata/bertweet-base-sentiment-analysis` (Twitter-optimized)
- `j-hartmann/emotion-english-distilroberta-base` (8 basic emotions) 