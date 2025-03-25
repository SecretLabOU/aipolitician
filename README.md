# AI Politician 🇺🇸

[![Mistral AI](https://img.shields.io/badge/Mistral--7B-Powered-blue)](https://mistral.ai/)
[![Fine-tuned](https://img.shields.io/badge/Custom-Fine--tuned-green)](https://huggingface.co/nnat03)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![GPU](https://img.shields.io/badge/CUDA-12.x-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

Chat with AI versions of political figures using fine-tuned Mistral-7B models that capture their unique speaking styles, policy positions, and personality traits. Backed by a Retrieval-Augmented Generation (RAG) system for fact-based responses.

## 🚀 Features

- **Authentic Personality Emulation**: Experience realistic conversations with political figures
- **Factual Enhancement**: Optional RAG system connects models to factual knowledge
- **Memory-Efficient**: Uses 4-bit quantization to run on consumer GPUs
- **Docker Ready**: Everything containerized for easy deployment
- **Dual Interfaces**: 
  - Simple direct chat mode for quick interactions
  - Advanced LangGraph mode for more sophisticated conversations

## 💾 Installation

### Quick Start with Docker

The easiest way to get started is with Docker:

```bash
# Clone the repository
git clone https://github.com/yourusername/aipolitician.git
cd aipolitician

# Configure environment variables
cp .env.example .env
# Edit .env with your Hugging Face token

# Start the application
docker-compose up -d
```

### Manual Installation

For local development or customization:

```bash
# Clone the repository
git clone https://github.com/yourusername/aipolitician.git
cd aipolitician

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings
```

## 🎮 Usage

### Basic Chat Mode

For a straightforward chat experience:

```bash
# Chat with Trump (default)
python chat.py

# Chat with Biden
python chat.py --persona biden

# Enable factual enhancement
python chat.py --persona biden --rag

# Adjust response parameters
python chat.py --max-length 256 --temperature 0.8
```

During chat, type `toggle rag` to enable/disable factual enhancement or `exit` to quit.

### Advanced LangGraph Mode

For more sophisticated conversations using the LangGraph agent:

```bash
python political_chat.py --persona trump --rag
```

### Available Options

| Option          | Description                               | Default |
| --------------- | ----------------------------------------- | ------- |
| `--persona`     | Which persona to use (`trump` or `biden`) | `trump` |
| `--rag`         | Enable factual enhancement                | `false` |
| `--max-length`  | Maximum length of responses               | `512`   |
| `--temperature` | Response randomness (0.0-1.0)             | `0.7`   |

## 🏗️ Project Structure

```
aipolitician/
│
├── chat.py                # Basic chat interface
├── political_chat.py      # Advanced LangGraph chat interface
├── requirements.txt       # Project dependencies
├── .env.example           # Environment variables template
├── Dockerfile             # Container definition
├── docker-compose.yml     # Multi-container orchestration
│
├── db/                    # RAG database components
│   └── utils/             # Database utilities
│       └── rag_utils.py   # RAG integration functions
│
└── lang-graph/            # LangGraph integration
    └── src/
        └── political_agent_graph/
            ├── config.py        # Agent configuration
            ├── graph.py         # Conversation flow
            ├── local_models.py  # Model management
            └── prompts.py       # System prompts
```

## 🔍 How It Works

1. **Model Architecture**: Uses Mistral-7B-Instruct fine-tuned with LoRA adapters
2. **Persona Emulation**: Custom fine-tuning on speeches, interviews, and policy documents
3. **Factual Enhancement**: Vector database with semantic search for retrieving relevant facts
4. **Memory Efficiency**: 4-bit quantization allows running on consumer GPUs (8GB+ VRAM)

## 📝 Troubleshooting

### Model Loading Problems

If you see errors loading models:

```
Error: Failed to load model: CUDA out of memory
```

**Solutions**:
- Use a smaller `--max-length` value (like 256)
- Check GPU memory with `nvidia-smi`
- Close other GPU-intensive applications
- Try adding `export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` before running

### RAG System Issues

If factual enhancement isn't working:

**Solutions**:
- Ensure Milvus database is running: `docker-compose ps`
- Check database logs: `docker-compose logs milvus`
- Restart the database stack: `docker-compose restart milvus etcd minio`

### GPU Not Detected

If CUDA isn't detected:

**Solutions**:
- Verify CUDA installation: `nvidia-smi`
- Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- Update GPU drivers

## 📋 Requirements

- **Hardware**: CUDA-compatible GPU with 8GB+ VRAM
- **Software**: 
  - Python 3.10+
  - CUDA 11.7+ and compatible drivers
  - Docker and Docker Compose (for containerized use)
- **Storage**: 20GB+ free disk space

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This project is for educational and research purposes. The AI models attempt to mimic the speaking styles and policy positions of public figures but do not represent their actual views or statements. Use responsibly.