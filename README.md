# AI Politician

Fine-tuned Mistral-7B models that emulate Donald Trump's and Joe Biden's speaking styles.

## Prerequisites

### CUDA Setup
- CUDA compatible GPU required
- CUDA toolkit and drivers must be installed
- Note: The project was tested with CUDA 12.4

### Environment Setup
This project requires two separate conda environments due to specific version requirements:

1. Training Environment:
```bash
conda create -n training-env python=3.10
conda activate training-env
pip install -r requirements-training.txt
```

2. Chat Environment:
```bash
conda create -n chat-env python=3.10
conda activate chat-env
pip install -r requirements-chat.txt
```

### HuggingFace Setup
1. Create a HuggingFace account and get your API key
2. The API key needs access to:
   - mistralai/Mistral-7B-Instruct-v0.2
   - Trump dataset (specific access requirements)

### Environment Variables
1. Create a `.env` file from the example:
```bash
cp .env.example .env
```

2. Configure your environment:
```
HUGGINGFACE_API_KEY=your_key_here
SHARED_MODELS_PATH=/path/to/models  # Default: /home/shared_models/aipolitician
```

## Dataset Setup

### Required Datasets
1. Biden Datasets:
   - Joe Biden tweets dataset
   - Joe Biden 2020 DNC speech
   Place in: `/home/natalie/datasets/biden/`
   - joe-biden-tweets.zip
   - joe-biden-2020-dnc-speech.zip

2. Trump Datasets:
   - Trump interviews dataset
   - Trump speeches dataset
   (Specific dataset locations and formats to be documented)

## Training

### Important Note
Use the `python` command for training scripts in the training environment:

```bash
conda activate training-env
python training/train_mistral_trump.py
python training/train_mistral_biden.py
```

The training process:
- Loads Mistral-7B-Instruct-v0.2 base model
- Fine-tunes using LoRA
- Uses 4-bit quantization for memory efficiency
- Saves models to:
  - `mistral-trump/` and `fine_tuned_trump_mistral/`
  - `mistral-biden/` and `fine_tuned_biden_mistral/`

## Chat Interface

### Important Note
Use the `python3` command for chat scripts in the chat environment:

```bash
conda activate chat-env
python3 chat_trump.py  # Chat with Trump AI
python3 chat_biden.py  # Chat with Biden AI
```

Each script provides:
- Interactive terminal interface
- Type messages and get responses in the respective style
- Type 'quit' to exit
- Use Ctrl+C to exit at any time

## Database RAG System

The project includes a Retrieval-Augmented Generation (RAG) database system that provides factual information to the AI models. This system improves factual accuracy by retrieving relevant information from a set of structured databases.

### Database Setup

1. Create the database directory:
```bash
mkdir -p /home/natalie/Databases/political_rag
```

2. Initialize the databases:
```bash
conda activate chat-env
python -m db.scripts.initialize_databases
```

### Using RAG in Chat

By default, the chat interfaces will use the RAG system if available. To disable RAG:

```bash
python3 chat_biden.py --no-rag
python3 chat_trump.py --no-rag
```

### Database Structure

The system includes 17 specialized databases:
- Biography Database
- Policy Database
- Voting Record Database
- Public Statements Database
- And many more...

For full details, see the [Database README](db/README.md).

## Project Structure

```
.
├── .env                      # Environment variables
├── .env.example              # Example environment file
├── requirements-training.txt # Training environment dependencies
├── requirements-chat.txt     # Chat environment dependencies
├── chat_biden.py             # Biden chat interface
├── chat_trump.py             # Trump chat interface
├── db/                       # Database RAG system
│   ├── config.py             # Database configuration
│   ├── database.py           # Base database interface
│   ├── README.md             # Database documentation
│   ├── schemas/              # Database schema definitions
│   ├── scripts/              # Database scripts
│   └── utils/                # Database utilities
└── training/                 # Training code
    ├── train_mistral_biden.py
    └── train_mistral_trump.py
```

## Version Compatibility

### Training Environment
- Specific versions required for training stability
- See requirements-training.txt for exact versions
- Key packages:
  - torch==2.0.1
  - transformers==4.36.0
  - peft==0.7.0

### Chat Environment
- More flexible version requirements
- See requirements-chat.txt for minimum versions
- Key packages:
  - torch>=2.1.0
  - transformers>=4.36.0
  - peft>=0.7.0

## Troubleshooting

### Common Issues
1. Version Conflicts
   - Ensure you're using the correct conda environment
   - Check that you're using the right python command (python for training, python3 for chat)
   - Verify package versions match requirements files

2. CUDA Issues
   - Verify CUDA is properly installed
   - Check GPU availability with `nvidia-smi`
   - Ensure CUDA version compatibility

3. Model Loading Issues
   - Verify model paths in .env file
   - Check HuggingFace API key permissions
   - Ensure all required model files are present

For additional help or to report issues, please open a GitHub issue.
