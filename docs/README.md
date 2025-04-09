# 📚 AI Politician Documentation

<p align="center">
  <img src="https://img.shields.io/badge/Python%203.9+-blue.svg" alt="Platform">
  <img src="https://img.shields.io/badge/Mistral--7B-Powered-green.svg" alt="Models">
  <img src="https://img.shields.io/badge/LangGraph-Workflow-orange.svg" alt="Framework">
</p>

This directory contains comprehensive documentation for the AI Politician system, explaining how the chat, debate, and knowledge retrieval systems work together.

## 📋 Documentation Index

### Core Systems

- [Chat System](chat.md) - The interactive chat interface with AI politicians
- [Debate System](debate.md) - The AI politician debate simulation system
- [RAG System](rag.md) - The Retrieval-Augmented Generation knowledge system

### Additional Resources

- [Installation Guide](installation.md) - Detailed setup instructions
- [Contributing Guide](contributing.md) - How to contribute to the project
- [Training Guide](training.md) - How to train new politician models

## 🚀 Quick Start

Get started with the AI Politician system quickly:

### Chat Mode

```bash
# Chat with Biden
python aipolitician.py chat biden

# Chat with Trump
python aipolitician.py chat trump

# Chat with debugging information
python aipolitician.py debug biden
```

### Debate Mode

```bash
# Run a default debate
python aipolitician.py debate

# Debate on a specific topic
python aipolitician.py debate --topic "Climate Change"

# Use a specific debate format
python aipolitician.py debate --format "town_hall"
```

### Running Without Knowledge Retrieval

For all modes, you can disable the RAG system if needed:

```bash
# Chat without RAG
python aipolitician.py chat biden --no-rag

# Debate without RAG
python aipolitician.py debate --no-rag
```

## 📦 System Architecture

The AI Politician system consists of three main components:

### 1. Chat System

An interactive interface for one-on-one conversations with AI politicians. Features include:

- **Context Agent**: Extracts topics and retrieves relevant knowledge
- **Sentiment Agent**: Analyzes query sentiment and handles sensitive questions
- **Response Agent**: Generates authentic-sounding politician responses
- **Multiple Output Modes**: Regular, debug, and trace modes

### 2. Debate System

A simulation of political debates between AI politicians. Key features:

- **Moderator Control**: An agent that manages the debate flow
- **Multiple Debate Formats**: Town hall, head-to-head, and panel discussions
- **Fact Checking**: Verification of factual claims made during debates
- **Topic Management**: Intelligent transition between debate subtopics

### 3. RAG System

The knowledge retrieval system that enhances responses with factual information:

- **ChromaDB Vector Database**: Stores and retrieves document embeddings
- **SentenceTransformer**: Creates vector representations for semantic search
- **Filtering System**: Retrieves politician-specific relevant information
- **Context Formatting**: Structures retrieved knowledge for model prompts

## 📊 Project Structure

The AI Politician project is organized as follows:

```
aipolitician/
├── aipolitician.py             # Unified entry point
├── scripts/
│   ├── chat/                   # Chat scripts
│   │   ├── chat_politician.py
│   │   ├── debug_politician.py
│   │   └── trace_politician.py
│   └── debate/                 # Debate scripts
│       ├── debate_politician.py
│       ├── debug_debate.py
│       └── test_debate_simple.py
├── src/
│   ├── models/                 # Core model implementations
│   │   ├── langgraph/          # LangGraph workflows
│   │   │   ├── agents/         # Agent implementations
│   │   │   └── debate/         # Debate-specific components
│   └── data/                   # Data management
│       └── db/                 # Database components
│           └── chroma/         # ChromaDB implementation
└── docs/                       # Documentation
    ├── chat.md
    ├── debate.md
    └── rag.md
```

## 🔧 Troubleshooting

If you encounter issues:

1. Check the specific documentation for the component you're using:
   - [Chat Troubleshooting](chat.md#troubleshooting)
   - [Debate Troubleshooting](debate.md#troubleshooting)
   - [RAG Troubleshooting](rag.md#troubleshooting)

2. Try running with the `--no-rag` flag to isolate knowledge retrieval issues:
   ```bash
   python aipolitician.py chat biden --no-rag
   ```

3. Use trace mode for detailed debugging information:
   ```bash
   python aipolitician.py trace biden
   ```

## 🤝 Contributing

We welcome contributions to improve the AI Politician system. See the [Contributing Guide](contributing.md) for details on how to get started. 