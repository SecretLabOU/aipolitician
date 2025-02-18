# AI Politician

An AI-powered application that simulates conversations with political figures, featuring a fine-tuned Mistral-7B model trained to emulate Donald Trump's speaking style.

## Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended)
- Conda or Miniconda
- HuggingFace API key with access to the Mistral-7B-Instruct-v0.2 model (gated model)

## Model Access

This project uses the Mistral-7B-Instruct-v0.2 model, which is a gated model on HuggingFace. To use this project:

1. Request access to the model at: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
2. Create a HuggingFace API key at: https://huggingface.co/settings/tokens
3. Add your API key to the .env file

## Shared Model Location

The fine-tuned model weights are stored in a shared location accessible to all users:
```
/home/shared_models/aipolitician/fine_tuned_trump_mistral/
```

This location is automatically configured in the application settings.

## Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd aipolitician
```

2. Create environment file:
```bash
cp .env.example .env
```

3. Edit .env and add your HuggingFace API key:
```
HUGGINGFACE_API_KEY=your_key_here
```

4. Run the application:
```bash
chmod +x run.sh
./run.sh
```

The server will start at http://localhost:8000

## API Endpoints

- `POST /chat/donald-trump` - Chat with Trump AI
- `GET /health` - Health check endpoint
- `DELETE /sessions/{session_id}` - End a chat session

## Training

The model training code is located in the `training/` directory. To train the model:

1. Ensure you have access to the Mistral-7B-Instruct-v0.2 model
2. Set up your HuggingFace API key in your environment
3. Run the training script:
```bash
python training/train_mistral_trump.py
```

## Project Structure

```
aipolitician/
├── app/
│   ├── agents/         # AI agent implementations
│   ├── models/         # Model configurations
│   └── utils/          # Utility functions
├── training/           # Model training code
└── .env.example        # Environment template
```

## Environment Variables

- `HUGGINGFACE_API_KEY`: Your HuggingFace API key (required)
- `SHARED_MODELS_PATH`: Path to shared models (default: /home/shared_models/aipolitician)
- `MAX_REQUEST_TOKENS`: Maximum tokens per request (default: 1000)
- `REQUEST_TIMEOUT`: Request timeout in seconds (default: 300)

## GPU Requirements

The application is optimized for GPU usage and will automatically:
- Detect available GPUs
- Configure appropriate worker counts
- Manage GPU memory efficiently
- Clean up resources when sessions are inactive

## Security Notes

- Keep your .env file secure and never commit it to version control
- The HuggingFace API key must have access to the Mistral model
- The application includes automatic session cleanup for security
