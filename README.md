# AI Politician 🇺🇸

[![Mistral AI](https://img.shields.io/badge/Mistral--7B-Powered-blue)](https://mistral.ai/)
[![Fine-tuned](https://img.shields.io/badge/Custom-Fine--tuned-green)](https://huggingface.co/nnat03)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![CUDA 12.4](https://img.shields.io/badge/CUDA-12.4-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

Fine-tuned Mistral-7B language models that emulate Donald Trump's and Joe Biden's speaking styles, discourse patterns, and policy positions. Talk to AI versions of both political figures using advanced NLP models enhanced with factual knowledge through a Retrieval-Augmented Generation (RAG) system.

## 🌟 Features

- **Personality-based Response Generation**: Chat with AI models fine-tuned to capture the distinct communication styles of Trump and Biden
- **RAG System Integration**: Ensures factual accuracy by retrieving real information from specialized political databases
- **Memory-Efficient Inference**: Optimized using 4-bit quantization for better performance on consumer hardware
- **Interactive Chat Interface**: Simple command-line interface for conversing with either political figure
- **ChromaDB Vector Database**: Semantic search capabilities for efficient information retrieval

## 📁 Project Structure

```
aipolitician/
├── aipolitician.py             # Unified launcher script
├── langgraph_politician.py    # Main LangGraph entry point
├── requirements/
│   ├── requirements-base.txt
│   ├── requirements-chat.txt
│   ├── requirements-langgraph.txt
│   ├── requirements-browser-fact-checker.txt
│   ├── requirements-training.txt
│   └── requirements-all.txt
├── scripts/
│   ├── chat/
│   │   ├── chat_politician.py
│   │   ├── debug_politician.py
│   │   └── trace_politician.py
│   ├── run_debate.py
│   ├── test_debate.py
│   └── test_debate_simple.py
├── src/
│   ├── __init__.py
│   └── models/
│       ├── __init__.py
│       ├── chat/
│       │   ├── __init__.py
│       │   ├── chat_biden.py
│       │   └── chat_trump.py
│       ├── langgraph/
│       │   ├── __init__.py
│       │   ├── agents/
│       │   │   ├── context_agent.py
│       │   │   ├── response_agent.py
│       │   │   └── sentiment_agent.py
│       │   ├── debate/
│       │   ├── utils/
│       │   ├── api.py
│       │   ├── cli.py
│       │   ├── config.py
│       │   └── workflow.py
│       └── training/
├── docs/
│   ├── README.md
│   ├── chat_system.md
│   ├── langgraph_workflow.md
│   ├── system_overview.md
│   └── usage_guide.md
├── .env.example
├── .gitignore
├── README.md
└── setup.py
```

## 📚 Documentation

Comprehensive documentation for key components is available in the `docs` folder:

- [System Overview](docs/system_overview.md) - High-level architecture and flow
- [Usage Guide](docs/usage_guide.md) - How to use the system
- [Chat System](docs/chat_system.md) - How to use the chat interface
- [LangGraph Workflow](docs/langgraph_workflow.md) - Details on the LangGraph implementation

## 🔄 Pretrained Models

The models are hosted on Hugging Face and can be accessed here:

- [Trump Model (nnat03/trump-mistral-adapter)](https://huggingface.co/nnat03/trump-mistral-adapter)
- [Biden Model (nnat03/biden-mistral-adapter)](https://huggingface.co/nnat03/biden-mistral-adapter)

These are LoRA adapters designed to be applied to the [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) base model.

## 🚀 Installation

### Prerequisites
- Python 3.8+ (recommended: Python 3.10)
- Conda (for environment management)
- CUDA 12.0+ (optional, for GPU acceleration)

### Option 1: Install from Source (Recommended)
```bash
# Clone the repository
git clone https://github.com/yourusername/aipolitician.git
cd aipolitician

# Create and activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install all dependencies

```
pip install -r requirements/requirements-base.txt -r requirements/requirements-chat.txt -r requirements/requirements-langgraph.txt
```

### Setting up the Database (for RAG features)

The AI Politician uses ChromaDB as its vector database for RAG features:

```bash
# Install ChromaDB dependencies
pip install -r requirements/requirements-langgraph.txt

# Initialize the database
cd src/data/db/chroma
./setup.sh
```

ChromaDB is used for retrieving relevant factual information to enhance the quality of responses.

For more details, see the [ChromaDB setup instructions](docs/data/chroma/setup_instructions.md).

## 💬 Usage

The system provides three ways to interact with the AI Politician:

### Unified Launcher

The easiest way to use the system is with the unified launcher:

```bash
# Clean chat mode
python aipolitician.py chat biden

# Debug mode
python aipolitician.py debug biden

# Trace mode
python aipolitician.py trace biden

# Disable RAG database (for any mode)
python aipolitician.py chat biden --no-rag
```

For more detailed usage instructions, see the [Chat System](docs/chat_system.md) documentation.

## System Components

1. **Context Agent**: Extracts topics from user input and retrieves relevant knowledge
2. **Sentiment Agent**: Analyzes the sentiment and decides if deflection is needed
3. **Response Agent**: Generates the final response using the politician's style

For more details on how these components work together, see the [LangGraph Workflow](docs/langgraph_workflow.md) documentation.

## 🔄 Adding New Politicians

To add a new politician to the system:

1. Collect training data for the politician
2. Process the data into the appropriate format
3. Fine-tune a model for the politician
4. Add the necessary chat model implementation
5. Update the relevant configuration files

For detailed instructions, see the [Model Training](docs/model_training.md) documentation.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [Mistral AI](https://mistral.ai/) for the base models
- [LangChain](https://www.langchain.com/) for the LangGraph framework
- [ChromaDB](https://www.trychroma.com/) for the vector database
- [PEFT](https://github.com/huggingface/peft) for efficient fine-tuning