# Simplified Lang-Graph with Docker

This is a simplified version of the lang-graph system that runs in Docker. It uses mock models to demonstrate the system's functionality without requiring large language models or complex integrations.

## Quick Start with Docker

```bash
# Build and run the container
cd lang-graph
docker-compose up --build
```

This will start the application in interactive mode with the Donald Trump persona.

## Alternative Ways to Run

### Run the Demo

To run the quick demo that showcases multiple politicians:

```bash
docker-compose run --rm lang-graph python src/main.py --demo
```

### List Available Politicians 

```bash
docker-compose run --rm lang-graph python src/main.py --list
```

### Run Specific Politician

```bash
docker-compose run --rm lang-graph python src/main.py joe_biden
```

## Without Docker

If you prefer not to use Docker, you can run the application directly:

```bash
# Setup dependencies
./setup_models.sh

# Run the application
./run.sh
```

## System Architecture

This simplified system demonstrates the lang-graph architecture without complex dependencies:

1. Uses LangGraph for agent-based conversations
2. Simulates politician personas with simplified models
3. Maintains the core conversation flow and graph structure
4. Eliminates dependencies on large language models

## Customization

You can modify the responses in `src/political_agent_graph/local_models.py` to customize how the politicians respond. 