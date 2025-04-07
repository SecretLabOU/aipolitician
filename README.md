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

The project has been organized into a clean, modular structure:

```
aipolitician/
├── aipolitician.py        # Unified launcher script
├── langgraph_politician.py # Main entry point
├── requirements/           # Dependencies
├── scripts/                # Helper scripts
│   ├── chat_politician.py  # Clean chat mode
│   ├── debug_politician.py # Debug mode with analysis info
│   ├── trace_politician.py # Trace mode with detailed output
│   └── manage_db.py        # Database management script
└── src/                    # Core source code
    ├── data/               # Data storage
    │   ├── db/             # Database implementation
    │   ├── scraper/        # Web scraper for data collection
    │   └── pipeline/       # Data processing pipeline
    └── models/             # Model definitions
        ├── chat/           # Chat models
        ├── langgraph/      # LangGraph implementation
        │   ├── agents/     # Individual agents
        │   │   ├── context_agent.py    # Context extraction
        │   │   ├── sentiment_agent.py  # Sentiment analysis
        │   │   └── response_agent.py   # Response generation
        │   ├── config.py   # Configuration settings
        │   ├── cli.py      # Command-line interface
        │   └── workflow.py # Workflow definition
        └── training/       # Training utilities
```

## 📚 Documentation

Comprehensive documentation for each component is available in the `docs` folder:

- [System Overview](docs/system_overview.md) - High-level architecture and flow
- [Chat System](docs/chat_system.md) - How to use the chat interface
- [LangGraph Workflow](docs/langgraph_workflow.md) - Details on the LangGraph implementation
- [Database System](docs/database_system.md) - Using the vector database for knowledge retrieval
- [Scraper System](docs/scraper_system.md) - Collecting data from various sources
- [Pipeline System](docs/pipeline_system.md) - Processing data for the knowledge database
- [Model Training](docs/model_training.md) - Training politician-specific models

## 🔗 Pretrained Models

The models are hosted on Hugging Face and can be accessed here:

- [Trump Model (nnat03/trump-mistral-adapter)](https://huggingface.co/nnat03/trump-mistral-adapter)
- [Biden Model (nnat03/biden-mistral-adapter)](https://huggingface.co/nnat03/biden-mistral-adapter)

These are LoRA adapters designed to be applied to the [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) base model.

## 🚀 Installation

### Prerequisites
- Python 3.8+ (recommended: Python 3.10)
- CUDA 12.0+ (for GPU acceleration)
- Docker and Docker Compose (for Milvus database)

### Option 1: Install from Source
```bash
# Clone the repository
git clone https://github.com/yourusername/aipolitician.git
cd aipolitician

# Create and activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install all dependencies
pip install -r requirements/requirements-base.txt
pip install -r requirements/requirements-chat.txt
pip install -r requirements/requirements-langgraph.txt
```

### Option 2: Install Specific Components
```bash
# Install only the chat interface dependencies
pip install -r requirements/requirements-chat.txt

# Install only the scraper dependencies 
pip install -r requirements/requirements-scraper.txt

# Install only the training dependencies
pip install -r requirements/requirements-training.txt
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
- [Milvus](https://milvus.io/) for the vector database
- [PEFT](https://github.com/huggingface/peft) for efficient fine-tuning