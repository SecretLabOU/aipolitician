# AI Politician <img src="https://img.shields.io/badge/USA-Official-blue.svg" width="80" align="center" alt="USA">

<p align="center">
  <img src="https://img.shields.io/badge/Mistral--7B-Powered-0066CC?style=for-the-badge&logo=mistral&logoColor=white" alt="Mistral AI">
  <img src="https://img.shields.io/badge/Fine--tuned-Models-34C759?style=for-the-badge&logo=huggingface&logoColor=white" alt="Fine-tuned">
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10">
  <img src="https://img.shields.io/badge/CUDA-12.4-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA 12.4">
</p>

<p align="center">
  <b>An advanced AI system for simulating political discourse through natural language processing</b>
</p>

---

## Overview

AI Politician is a cutting-edge research project leveraging fine-tuned Mistral-7B language models to faithfully recreate the speaking styles, discourse patterns, and policy positions of prominent political figures. The system creates an immersive experience for users to engage with AI representations through:

- **Interactive Conversations** — Direct one-on-one dialogues with political figures
- **Dynamic Debates** — Realistic simulations of political discussions on contemporary issues
- **Fact-Validated Responses** — Knowledge retrieval through advanced RAG technology

## Key Features

<table>
  <tr>
    <td width="33%" align="center"><b>💬 Personality Modeling</b></td>
    <td width="33%" align="center"><b>🎤 Debate Simulation</b></td>
    <td width="33%" align="center"><b>🔍 Knowledge Retrieval</b></td>
  </tr>
  <tr>
    <td>Distinctive communication styles accurately reflect each politician's unique voice and rhetorical patterns</td>
    <td>Sophisticated moderator-controlled debates with fact-checking and natural topic progression</td>
    <td>Real-time information access ensures responses are grounded in factual context</td>
  </tr>
</table>

## Architecture

The system consists of three integrated components:

<table>
  <tr>
    <td width="33%" align="center"><b>Chat System</b></td>
    <td width="33%" align="center"><b>Debate System</b></td>
    <td width="33%" align="center"><b>RAG System</b></td>
  </tr>
  <tr>
    <td>Topic extraction, sentiment analysis, and personalized response generation</td>
    <td>Structured format control, cross-examination, and balanced speaking time</td>
    <td>Vector search, contextual filtering, and knowledge integration</td>
  </tr>
</table>

## Technology Stack

- **Foundation**: [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) base models
- **Personalization**: Custom LoRA adapters with PEFT for efficient fine-tuning
- **Orchestration**: LangChain and LangGraph for complex workflow management
- **Knowledge Base**: ChromaDB vector database with SentenceTransformers embeddings
- **Performance**: 4-bit quantization enabling efficient inference on consumer hardware

## Quick Start

```bash
# Chat with Biden
python aipolitician.py chat biden

# Chat with Trump
python aipolitician.py chat trump

# Run a moderated debate
python aipolitician.py debate --topic "Climate Change"
```

## Pretrained Models

Access our fine-tuned politician models on Hugging Face:

- [Biden Model](https://huggingface.co/nnat03/biden-mistral-adapter) — Emulates President Biden's communication style
- [Trump Model](https://huggingface.co/nnat03/trump-mistral-adapter) — Captures former President Trump's distinctive rhetoric

## System Requirements

- **Python**: 3.9+ (3.10 recommended)
- **GPU**: CUDA 12.0+ (optional, significantly improves performance)
- **Storage**: 8GB minimum for models and vector database

## Documentation

Comprehensive guides available in the `docs` directory:

- [Chat System](docs/chat.md) — Interactive dialogue interface
- [Debate System](docs/debate.md) — Multi-agent debate simulation
- [RAG System](docs/rag.md) — Knowledge retrieval implementation

## License

This project is available under the MIT License. See the LICENSE file for details.

## Acknowledgements

We extend our appreciation to:

- [Mistral AI](https://mistral.ai/) for their exceptional base models
- [LangChain](https://www.langchain.com/) for the LangGraph framework
- [ChromaDB](https://www.trychroma.com/) for vector database technology
- [SentenceTransformers](https://www.sbert.net/) for embedding models
- [PEFT](https://github.com/huggingface/peft) for efficient fine-tuning methods