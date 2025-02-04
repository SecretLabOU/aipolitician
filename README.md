# PoliticianAI

An AI system for political discourse simulation using LangChain and open-source models. This system enables users to interact with virtual agents representing political figures, leveraging voting history, policy positions, and public statements to generate informed responses.

## Features

- Sentiment analysis of user queries
- Context-aware topic classification
- Policy position retrieval
- Voting record analysis
- Dynamic response generation
- Fact-based responses
- Conversation memory
- GPU acceleration support

## Data Sources

The system uses data from:
- Hugging Face datasets for speeches and statements
  * bananabot/TrumpSpeeches
  * yunfan-y/trump-tweets-cleaned
  * pookie3000/trump-interviews
- Congress.gov public data for voting records
- FEC API for campaign finance data (optional)
- Sample data for demonstration purposes

## Architecture

### Core Components

1. **Agents**
   - SentimentAgent: Analyzes emotional tone using DistilBERT
   - ContextAgent: Extracts topics using BART
   - ResponseAgent: Generates responses using LLaMA 2

2. **Data Storage**
   - SQLite database for structured data
   - FAISS vector store for embeddings
   - File-based cache for performance

3. **Models**
   - LLaMA 2 (7B) for response generation
   - DistilBERT for sentiment analysis
   - BART for context understanding

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM
- 50GB+ disk space

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/PoliticianAI.git
cd PoliticianAI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Download required models:
```bash
python scripts/setup_models.py
```

5. Initialize data:
```bash
python scripts/collect_data.py
```

## Project Structure

```
politicianai/
├── src/
│   ├── agents/          # AI Agents
│   ├── database/        # Database models
│   ├── config.py        # Configuration
│   └── utils/           # Utilities
├── data/
│   ├── raw/             # Raw collected data
│   ├── processed/       # Processed data
│   └── embeddings/      # FAISS indices
├── models/              # AI models
├── scripts/             # Utility scripts
└── tests/              # Test suite
```

## Usage

1. Start the server:
```bash
python main.py
```

2. Access the API at `http://localhost:8000`

3. Available endpoints:
   - POST `/chat`: Send user queries
   - GET `/topics`: List available topics
   - GET `/conversation_history`: Get chat history
   - POST `/clear_history`: Clear chat history
   - GET `/health`: Check system status

Example chat request:
```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"text": "What is John Smith'\''s position on healthcare?"}'
```

## Development

1. Set up development environment:
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

## Docker Support

1. Build and run with Docker:
```bash
docker-compose up --build
```

2. Development mode:
```bash
docker-compose --profile dev up
```

3. With monitoring:
```bash
docker-compose --profile monitoring up
```

## Monitoring

Access monitoring dashboards:
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090

## GPU Support

The system automatically detects and uses available GPUs. For optimal performance:
- CUDA 11.4+
- 8GB+ VRAM
- Latest GPU drivers

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- LangChain for the agent framework
- Hugging Face for transformer models and datasets
- Meta for LLaMA 2
- FastAPI for the web framework

## Future Improvements

1. Enhanced Features
   - Multi-politician comparisons
   - Timeline analysis
   - Real-time data updates
   - Multi-language support

2. Technical Improvements
   - Distributed processing
   - Advanced caching
   - Model quantization
   - API authentication

## Troubleshooting

Common issues and solutions:

1. GPU Memory Issues
   - Reduce batch sizes
   - Use model quantization
   - Free unused memory

2. Database Errors
   - Check permissions
   - Verify schema migrations
   - Clear corrupt cache

3. Model Loading
   - Verify downloads
   - Check CUDA compatibility
   - Update drivers

For more help, check the issues section in the repository.
