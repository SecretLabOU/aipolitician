# AI Politician - Trump Model

A fine-tuned Mistral-7B model that emulates Donald Trump's speaking style.

## Setup

1. Create a `.env` file from the example:
```bash
cp .env.example .env
```

2. Add your HuggingFace API key to `.env`:
```
HUGGINGFACE_API_KEY=your_key_here
SHARED_MODELS_PATH=/path/to/models
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training

To train the model:
```bash
python training/train_mistral_trump.py
```

This will:
- Load the Mistral-7B-Instruct-v0.2 base model
- Fine-tune it on Trump's speeches and interviews
- Save the resulting model to the path specified in SHARED_MODELS_PATH

## Chat Interface

To chat with the model:
```bash
python chat.py
```

This provides an interactive terminal interface where you can:
- Type messages and get responses in Trump's style
- Type 'quit' to exit
- Use Ctrl+C to exit at any time

## Project Structure

```
.
├── .env                    # Environment variables
├── .env.example           # Example environment file
├── requirements.txt       # Python dependencies
├── chat.py               # Interactive chat interface
└── training/             # Training code
    └── train_mistral_trump.py
```

## Model Details

- Base Model: Mistral-7B-Instruct-v0.2
- Training Data:
  - Trump interviews dataset
  - Trump speeches dataset
- Uses LoRA for efficient fine-tuning
- 4-bit quantization for reduced memory usage
