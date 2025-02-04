# PoliticianAI

An AI system for political discourse simulation where users interact with virtual agents representing political figures. The agents have knowledge of voting history, funding, public statements, and sentiment data, and can engage in dynamic debates to provide insightful discussions.

## Features

- Sentiment analysis of user input
- Context and topic extraction
- Dynamic response generation
- Historical data integration
- Real-time fact checking
- Caching for performance
- GPU acceleration support

## Architecture

### Core Components

1. **Agents**
   - `SentimentAgent`: Analyzes sentiment in user messages
   - `ContextAgent`: Extracts context and identifies topics
   - `ResponseAgent`: Generates contextual responses
   - `WorkflowManager`: Orchestrates agent interactions

2. **Database Models**
   - Topics
   - Politicians
   - Statements
   - Voting Records
   - Chat History
   - Response Cache

3. **API**
   - FastAPI-based REST endpoints
   - WebSocket support for real-time chat
   - Request/response validation
   - Error handling

4. **Utilities**
   - Caching system
   - Metrics tracking
   - Logging configuration
   - Helper functions

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM
- 50GB+ disk space
- Rust (will be installed automatically if needed)
- Miniconda or Anaconda

## Installation

1. SSH into the GPU server:
```bash
# For small jobs
ssh <username>

# For large jobs
ssh <username>
```

2. Clone and prepare the repository:
```bash
git clone https://github.com/yourusername/PoliticianAI.git
cd PoliticianAI
```

3. Initialize genv shell environment:
```bash
# Run the initialization script
./scripts/init_genv.sh

# Source your bashrc to apply changes
source ~/.bashrc
```

4. Run the GPU setup script with conda environment:
```bash
# If you want to use an existing conda environment:
./scripts/run_on_gpu.sh existing-env-name your-session-name

# Or to create a new conda environment:
./scripts/run_on_gpu.sh politician-ai your-session-name
```

The script will:
- Initialize genv shell environment
- Check GPU availability
- Set up/activate conda environment
- Install Rust if needed (required for ChromaDB)
- Install dependencies
- Download required models
- Initialize data
- Start the application

## Project Structure

```
PoliticianAI/
├── src/
│   ├── agents/           # AI agent implementations
│   ├── api/             # FastAPI endpoints
│   ├── database/        # SQLAlchemy models
│   └── utils/           # Helper utilities
├── scripts/
│   ├── init_genv.sh     # GPU environment setup
│   ├── run_on_gpu.sh    # Main run script
│   └── cleanup_gpu.sh   # GPU cleanup script
├── data/                # Data storage
│   ├── raw/             # Raw data files
│   ├── processed/       # Processed data
│   └── embeddings/      # Vector embeddings
├── models/              # Downloaded models
├── tests/               # Test suite
└── monitoring/          # Monitoring configs
```

## Usage

1. Start the server:
```bash
./scripts/run_on_gpu.sh politician-ai my-session
```

2. Access the API:
```bash
curl http://localhost:8000/health  # Health check
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is your stance on healthcare?"}'
```

3. When done, clean up:
```bash
./scripts/cleanup_gpu.sh
```

## API Endpoints

- `POST /chat`: Send message and get response
- `GET /topics`: List available topics
- `GET /politicians`: List available politicians
- `GET /health`: Health check endpoint

## Development

1. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

2. Run tests:
```bash
pytest tests/
```

3. Format code:
```bash
black src/ tests/
isort src/ tests/
```

4. Type checking:
```bash
mypy src/
```

## Monitoring

The system includes:
- Request/response metrics
- Model inference tracking
- Cache performance monitoring
- Database query metrics
- GPU utilization tracking

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
