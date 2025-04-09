# AI Politician 🇺🇸

[![Mistral AI](https://img.shields.io/badge/Mistral--7B-Powered-blue)](https://mistral.ai/)
[![Fine-tuned](https://img.shields.io/badge/Custom-Fine--tuned-green)](https://huggingface.co/nnat03)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![CUDA 12.4](https://img.shields.io/badge/CUDA-12.4-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

## Project Overview

AI Politician is a research project featuring fine-tuned Mistral-7B language models that emulate Donald Trump's and Joe Biden's speaking styles, discourse patterns, and policy positions. The system enables users to interact with AI versions of both political figures through:

- Direct chat conversations
- Simulated debates on political topics
- Fact-checking through a Retrieval-Augmented Generation (RAG) system

## Technologies

- **Base Models**: [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- **Fine-tuning**: Custom LoRA adapters using PEFT
- **Language Framework**: LangChain and LangGraph for workflow orchestration
- **RAG System**: ChromaDB vector database with SentenceTransformers
- **Optimization**: 4-bit quantization for efficient inference

## Key Features

- **Personality-based Response Generation**: Distinct communication styles of Trump and Biden
- **Debate System**: Moderated debates with fact-checking
- **RAG Integration**: Real-time knowledge retrieval for factual accuracy
- **Interactive Interface**: Simple command-line tools for different interaction modes

## Pretrained Models

The models are hosted on Hugging Face:

- [Trump Model (nnat03/trump-mistral-adapter)](https://huggingface.co/nnat03/trump-mistral-adapter)
- [Biden Model (nnat03/biden-mistral-adapter)](https://huggingface.co/nnat03/biden-mistral-adapter)

These are LoRA adapters designed to be applied to the Mistral-7B-Instruct-v0.2 base model.

## Documentation

For detailed information about using the system, please refer to the documentation in the `docs` folder:

- [Chat System](docs/chat.md) - Using the AI Politician chat interface
- [Debate System](docs/debate.md) - Running debates between AI politicians
- [RAG System](docs/rag.md) - Understanding the knowledge retrieval implementation

## Prerequisites

- Python 3.9+ (recommended: Python 3.10)
- CUDA 12.0+ (optional, for GPU acceleration)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Mistral AI](https://mistral.ai/) for the base models
- [LangChain](https://www.langchain.com/) for the LangGraph framework
- [ChromaDB](https://www.trychroma.com/) for the vector database
- [SentenceTransformers](https://www.sbert.net/) for embedding models
- [PEFT](https://github.com/huggingface/peft) for efficient fine-tuning