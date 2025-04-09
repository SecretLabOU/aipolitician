# AI Politician 🇺🇸

[![Mistral AI](https://img.shields.io/badge/Mistral--7B-Powered-blue)](https://mistral.ai/)
[![Fine-tuned](https://img.shields.io/badge/Custom-Fine--tuned-green)](https://huggingface.co/nnat03)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![CUDA 12.4](https://img.shields.io/badge/CUDA-12.4-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

Fine-tuned Mistral-7B language models that emulate Donald Trump's and Joe Biden's speaking styles, discourse patterns, and policy positions. Talk to AI versions of both political figures using advanced NLP models enhanced with factual knowledge through a Retrieval-Augmented Generation (RAG) system.

## 🌟 Features

- **Personality-based Response Generation**: Chat with AI models fine-tuned to capture the distinct communication styles of Trump and Biden
- **Debate System**: Watch AI politicians debate each other on various political topics with fact-checking
- **RAG System Integration**: Ensures factual accuracy by retrieving real information from specialized political databases
- **Memory-Efficient Inference**: Optimized using 4-bit quantization for better performance on consumer hardware
- **Interactive Chat Interface**: Simple command-line interface for conversing with either political figure
- **LangGraph Workflow**: Structured agent-based architecture for sophisticated interaction handling

## 📁 Project Structure

```
aipolitician/
├── aipolitician.py             # Unified launcher script (main entry point)
├── langgraph_politician.py     # Advanced LangGraph functionality
├── requirements/
│   ├── requirements-base.txt
│   ├── requirements-chat.txt
│   ├── requirements-debate.txt
│   ├── requirements-rag.txt
│   └── requirements-all.txt
├── scripts/
│   ├── chat/                   # Chat-related scripts
│   │   ├── chat_politician.py
│   │   ├── debug_politician.py
│   │   └── trace_politician.py
│   ├── debate/                 # Debate-related scripts
│   │   ├── debate_politician.py
│   │   ├── debug_debate.py
│   │   └── test_debate_simple.py
│   └── backup/                 # Legacy scripts
├── src/
│   ├── models/
│       ├── chat/
│       ├── langgraph/
│       │   ├── agents/
│       │   │   ├── context_agent.py
│       │   │   ├── response_agent.py
│       │   │   └── sentiment_agent.py
│       │   ├── debate/
│       │   │   ├── agents.py
│       │   │   ├── workflow.py
│       │   │   └── cli.py
│       │   ├── workflow.py
│   ├── data/
│       └── db/
│           └── chroma/         # ChromaDB configuration
├── docs/
│   ├── chat.md                 # Chat system documentation
│   ├── debate.md               # Debate system documentation
│   ├── rag.md                  # RAG system documentation
│   └── README.md               # Documentation index
└── README.md
```

## 📚 Documentation

Comprehensive documentation for key components is available in the `docs` folder:

- [Chat System](docs/chat.md) - How to use the AI Politician chat system
- [Debate System](docs/debate.md) - How to run debates between AI politicians
- [RAG System](docs/rag.md) - Details on the Retrieval-Augmented Generation implementation

## 🔄 Pretrained Models

The models are hosted on Hugging Face and can be accessed here:

- [Trump Model (nnat03/trump-mistral-adapter)](https://huggingface.co/nnat03/trump-mistral-adapter)
- [Biden Model (nnat03/biden-mistral-adapter)](https://huggingface.co/nnat03/biden-mistral-adapter)

These are LoRA adapters designed to be applied to the [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) base model.

## 🚀 Installation

### Prerequisites
- Python 3.9+ (recommended: Python 3.10)
- CUDA 12.0+ (optional, for GPU acceleration)

### Option 1: Basic Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/aipolitician.git
cd aipolitician

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install core dependencies
pip install -r requirements/requirements-base.txt
```

### Option 2: Full Installation with All Features
```bash
# Install all dependencies
pip install -r requirements/requirements-all.txt

# Set up the ChromaDB directory (for RAG)
sudo mkdir -p /opt/chroma_db
sudo chown $USER:$USER /opt/chroma_db
```

## 💬 Usage

The system provides a unified command-line interface for all functionality:

### Chat with a Politician

```bash
# Standard chat mode
python aipolitician.py chat biden

# Debug mode with more information
python aipolitician.py debug biden

# Detailed trace mode
python aipolitician.py trace biden

# Chat without RAG knowledge retrieval
python aipolitician.py chat trump --no-rag
```

### Run a Debate Between Politicians

```bash
# Basic debate with default settings (Biden vs Trump)
python aipolitician.py debate

# Debate on a specific topic
python aipolitician.py debate --topic "Climate Change"

# Debate with a specific format
python aipolitician.py debate --format "town_hall"

# Debate without RAG
python aipolitician.py debate --no-rag
```

## 🧠 System Architecture

### Chat System Components

1. **Context Agent**: Extracts topics from user input and retrieves relevant knowledge
2. **Sentiment Agent**: Analyzes the sentiment and decides if deflection is needed
3. **Response Agent**: Generates the final response using the politician's style

### Debate System Components

1. **Moderator Agent**: Controls the debate flow and manages transitions
2. **Politician Agents**: Generate responses for each politician
3. **Fact Checker**: Verifies factual claims made during the debate
4. **Topic Manager**: Manages debate topics and subtopics

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [Mistral AI](https://mistral.ai/) for the base models
- [LangChain](https://www.langchain.com/) for the LangGraph framework
- [ChromaDB](https://www.trychroma.com/) for the vector database
- [SentenceTransformers](https://www.sbert.net/) for embedding models
- [PEFT](https://github.com/huggingface/peft) for efficient fine-tuning