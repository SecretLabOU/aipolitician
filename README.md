# PoliticianAI

An AI system for dynamic political discourse simulation using virtual agents representing political figures.

## Project Structure

```
/
├── src/                 # Core application code
│   ├── agents/         # AI Agent implementations
│   ├── api/           # API endpoints
│   ├── database/      # Database models and operations
│   ├── utils/         # Utility functions
│   └── config.py      # Configuration settings
├── scripts/           # Utility scripts
│   ├── cleanup_gpu.sh           # Cleanup utility
│   ├── collect_politician_data.py # Data collection
│   ├── init_database.sh        # Database initialization
│   ├── init_genv.sh           # GPU environment setup
│   ├── run_on_gpu.sh         # GPU execution
│   ├── setup_models.py       # Model setup
│   └── update_database.sh    # Database updates
├── migrations/        # Database migrations
├── monitoring/       # System monitoring
│   ├── alerts.yml
│   ├── prometheus.yml
│   └── grafana/
└── tests/           # Testing suite
```

## Setup

1. Create and configure environment:
```bash
cp .env.example .env
# Edit .env with your settings
```

2. Initialize database:
```bash
./scripts/init_database.sh <conda-env-name>
```

3. Update database (when needed):
```bash
./scripts/update_database.sh <conda-env-name>
```

## GPU Usage

1. Run on GPU:
```bash
./scripts/run_on_gpu.sh <conda-env-name> <session-name>
```

2. Clean up GPU session:
```bash
./scripts/cleanup_gpu.sh <session-name>
```

## Data Collection

Collect politician data:
```bash
./scripts/collect_politician_data.py
```

## Configuration

Key configuration files:
- `.env`: Environment variables
- `src/config.py`: Application settings
- `alembic.ini`: Database migration settings
- `monitoring/`: Monitoring configuration

## Testing

Run tests:
```bash
pytest
```

## Monitoring

1. Metrics: Available through Prometheus
2. Dashboards: Available through Grafana
3. Logs: Structured logging with JSON format

## Key Components

### Agents
- Base Agent: Common agent functionality
- Context Agent: Handles conversation context
- Response Agent: Generates responses
- Sentiment Agent: Analyzes sentiment
- Workflow Manager: Orchestrates interactions

### Database
- Politicians: Political figure information
- Statements: Public statements and positions
- Topics: Political topics and categories
- Votes: Voting records
- Chat History: Interaction history

### Features
- Real-time data collection
- Sentiment analysis
- Topic categorization
- Dynamic response generation
- GPU acceleration
- Monitoring and metrics

## Requirements

See `requirements.txt` for full list of dependencies.

## License

[License details]
