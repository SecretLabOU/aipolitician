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
- **Milvus Vector Database**: Semantic search capabilities for efficient information retrieval

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
    │   └── db/             # Database files
    │       └── milvus/     # Vector database
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

See [docs/README.md](docs/README.md) for detailed information about the project structure.

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

# Install the package and all dependencies
pip install -e ".[scraper,training,chat]"
```

### Option 2: Install Specific Components
```bash
# Install only the chat interface dependencies
pip install -e ".[chat]"

# Install only the scraper dependencies
pip install -e ".[scraper]"

# Install only the training dependencies
pip install -e ".[training]"
```

### Setting up the Database (for RAG features)
```bash
# Create database directories
mkdir -p /home/username/Databases/ai_politician_milvus/data
mkdir -p /home/username/Databases/ai_politician_milvus/etcd
mkdir -p /home/username/Databases/ai_politician_milvus/minio

# Set up Milvus using Docker
cd src/data/db/milvus
./setup.sh

# Initialize the database schema
python scripts/initialize_db.py --recreate
```

## 💬 Usage

The system provides three ways to interact with the AI Politician:

### Unified Launcher

The easiest way to use the system is with the unified launcher:

```bash
# Clean chat mode
./aipolitician.py chat biden

# Debug mode
./aipolitician.py debug biden

# Trace mode
./aipolitician.py trace biden

# Disable RAG database (for any mode)
./aipolitician.py chat biden --no-rag
```

### Individual Scripts

You can also use the individual scripts directly:

#### 1. Clean Chat Mode

For a normal chat experience without technical details:

```bash
./scripts/chat_politician.py biden
# or
./scripts/chat_politician.py trump
```

#### 2. Debug Mode

For a chat with additional debugging information:

```bash
./scripts/debug_politician.py biden
# or
./scripts/debug_politician.py trump
```

#### 3. Trace Mode

For a detailed view of the entire workflow process:

```bash
./scripts/trace_politician.py biden
# or
./scripts/trace_politician.py trump
```

### Advanced Usage

You can also use the main script directly with more options:

```bash
python langgraph_politician.py chat --identity biden [--debug] [--trace] [--no-rag]
```

## Core Components

1. **Context Agent**: Extracts topics from user input and retrieves relevant knowledge
2. **Sentiment Agent**: Analyzes the sentiment and decides if deflection is needed
3. **Response Agent**: Generates the final response using the politician's style

## Database

The system uses Milvus for vector storage and retrieval when the `--no-rag` flag is not specified. If Milvus is not available, it falls back to synthetic responses.

## 🗄️ Vector Database System

The project includes a Retrieval-Augmented Generation (RAG) database system that provides factual information to enhance model responses. This system is built on Milvus, a powerful vector database.

### Database Features
- **Vector Similarity Search**: Find relevant information using semantic similarity
- **Schema Flexibility**: Combine structured data with vector embeddings
- **HNSW Indexing**: High-performance approximate nearest neighbor search
- **768-Dimensional Embeddings**: Using all-MiniLM-L6-v2 sentence transformer model

### Database Schema
The political figures collection contains comprehensive information including:
- Biographical details
- Policy positions
- Legislative records
- Public statements
- Timeline events
- Campaign history
- Personal information

For detailed database documentation, see [src/data/db/milvus/README.md](src/data/db/milvus/README.md).

## 🔄 Fine-tuning Process

The models were fine-tuned using the following datasets:

### Trump Model
- [Trump interviews dataset](https://huggingface.co/datasets/pookie3000/trump-interviews)
- [Trump speeches dataset](https://huggingface.co/datasets/bananabot/TrumpSpeeches)

### Biden Model
- [Biden tweets dataset (2007-2020)](https://www.kaggle.com/datasets/rohanrao/joe-biden-tweets)
- [Biden 2020 DNC speech dataset](https://www.kaggle.com/datasets/christianlillelund/joe-biden-2020-dnc-speech)

Place datasets in: `/home/natalie/datasets/biden/`
- joe-biden-tweets.zip
- joe-biden-2020-dnc-speech.zip

The training process:
- Used LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- Applied 4-bit quantization for memory efficiency
- Fine-tuned for 3 epochs with a cosine learning rate schedule
- Used a special instruction format to guide stylistic emulation

To run your own fine-tuning:
```bash
# Activate your virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Run the training scripts
python -m src.models.training.train_mistral_trump
python -m src.models.training.train_mistral_biden
```

## 🚧 Troubleshooting

### Common Issues

#### 1. Model Loading Issues
- **Symptom**: `Error loading model` or CUDA out-of-memory errors
- **Solutions**:
  - Ensure you have sufficient GPU memory
  - Verify CUDA is properly installed (`nvidia-smi`)
  - Check your HuggingFace API key has necessary permissions

#### 2. Database Connection Issues
- **Symptom**: `Failed to connect to Milvus server`
- **Solutions**:
  - Ensure Docker is running
  - Check if Milvus container is active: `docker ps | grep milvus`
  - Restart the database: `./src/data/db/milvus/setup.sh`

#### 3. Environment Conflicts
- **Symptom**: Import errors or version conflicts
- **Solutions**:
  - Make sure you're in the correct virtual environment
  - Ensure you installed the package with the right extras
  - Try reinstalling with `pip install -e ".[scraper,training,chat]"`

#### 4. Out-of-Memory Errors During Training
- **Symptom**: CUDA out-of-memory errors during training
- **Solutions**:
  - Reduce batch size in training scripts
  - Increase gradient accumulation steps
  - Use a GPU with more VRAM

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Mistral AI](https://mistral.ai/) for the base Mistral-7B model
- [Hugging Face](https://huggingface.co/) for hosting the models and datasets
- [Milvus](https://milvus.io/) for the vector database technology
- The open-source NLP and AI community

---

*Disclaimer: This project is created for educational and research purposes. The AI models attempt to mimic the speaking styles of public figures but do not represent their actual views or statements. Use responsibly.*

## Database Management

The system uses Milvus as a vector database for RAG. Use the database management script to control it:

```bash
# Start the database (cleans up any conflicting containers first)
./scripts/manage_db.py start

# Check database status
./scripts/manage_db.py status

# Load data into the database (first time setup)
./scripts/manage_db.py load

# Stop the database
./scripts/manage_db.py stop

# Restart the database
./scripts/manage_db.py restart
```