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

## 🔗 Pretrained Models

The models are hosted on Hugging Face and can be accessed here:

- [Trump Model (nnat03/trump-mistral-adapter)](https://huggingface.co/nnat03/trump-mistral-adapter)
- [Biden Model (nnat03/biden-mistral-adapter)](https://huggingface.co/nnat03/biden-mistral-adapter)

These are LoRA adapters designed to be applied to the [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) base model.

## 📋 Prerequisites

### Hardware Requirements
- CUDA-compatible GPU (8GB+ VRAM recommended)
- 16GB+ system RAM
- 20GB+ free disk space

### Software Requirements
- CUDA Toolkit and drivers (tested with CUDA 12.4)
- Python 3.10
- Conda package manager
- Docker (for database functionality)

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/aipolitician.git
cd aipolitician
```

### 2. Environment Setup
This project requires two separate conda environments due to specific version requirements:

#### Training Environment
```bash
conda create -n training-env python=3.10
conda activate training-env
pip install -r requirements-training.txt
```

#### Chat Environment
```bash
conda create -n chat-env python=3.10
conda activate chat-env
pip install -r requirements-chat.txt
```

### 3. Hugging Face API Setup
1. Create a [Hugging Face account](https://huggingface.co/join)
2. Generate an API key from your [settings page](https://huggingface.co/settings/tokens)
3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your Hugging Face API key
```

### 4. RAG Database Setup (Optional)
For enhanced factual responses, set up the Milvus vector database:

```bash
# Create database directories
mkdir -p /home/natalie/Databases/ai_politician_milvus/data
mkdir -p /home/natalie/Databases/ai_politician_milvus/etcd
mkdir -p /home/natalie/Databases/ai_politician_milvus/minio

# Set up Milvus using Docker
cd db/milvus
./setup.sh

# Initialize the database schema
python scripts/initialize_db.py --recreate
```

## 💬 Usage

### Chatting with Trump AI
```bash
conda activate chat-env
python3 chat_trump.py  # Basic mode
python3 chat_trump.py --rag  # With RAG for factual responses
```

### Chatting with Biden AI
```bash
conda activate chat-env
python3 chat_biden.py  # Basic mode
python3 chat_biden.py --rag  # With RAG for factual responses
```

### Command-Line Options
- `--rag`: Enable Retrieval-Augmented Generation for factual accuracy
- `--max-length INT`: Set maximum response length (default: 512 tokens)

### Example Questions
- "What's your plan for border security?"
- "How would you handle trade with China?"
- "Tell me about your healthcare policy."
- "What was your position on the Paris Climate Agreement?"
- "How would you address inflation?"

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

For detailed database documentation, see [db/milvus/README.md](db/milvus/README.md).

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
conda activate training-env
python training/train_mistral_trump.py
python training/train_mistral_biden.py
```

## 📂 Project Structure

```
.
├── .env                      # Environment variables
├── .env.example              # Example environment file
├── requirements-training.txt # Training environment dependencies
├── requirements-chat.txt     # Chat environment dependencies
├── chat_biden.py             # Biden chat interface
├── chat_trump.py             # Trump chat interface
├── db/                       # Database RAG system
│   ├── milvus/               # Milvus vector database
│   │   ├── scripts/          # Database initialization and search scripts
│   │   ├── docker-compose.yml # Docker configuration
│   │   ├── setup.sh          # Database setup script
│   │   └── cleanup.sh        # Database cleanup script
├── test/                     # Test scripts
│   ├── test_biden_model.py   # Test Biden model loading
│   ├── test_db.py            # Test database functionality
│   └── test_trump_model.py   # Test Trump model loading
└── training/                 # Training code
    ├── train_mistral_biden.py
    └── train_mistral_trump.py
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
  - Restart the database: `./db/milvus/setup.sh`

#### 3. Environment Conflicts
- **Symptom**: Import errors or version conflicts
- **Solutions**:
  - Make sure you're in the correct conda environment
  - Ensure you used the right requirements file
  - For training issues, use `python` command
  - For chat issues, use `python3` command

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