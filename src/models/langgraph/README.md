# AI Politician LangGraph System

This directory contains a LangGraph-based implementation of the AI Politician system, which simulates political figures (currently Biden and Trump) with realistic interactions, contextual awareness, and sentiment-based response modulation.

## System Architecture

The system uses LangGraph to orchestrate a workflow of specialized agents:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│   User Input    │────►│ Context Agent   │────►│ Sentiment Agent │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │                        │
                               ▼                        ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │                 │     │                 │
                        │  Knowledge DB   │     │   Sentiment     │
                        │  (RAG System)   │     │   Analysis      │
                        │                 │     │                 │
                        └─────────────────┘     └─────────────────┘
                               │                        │
                               └───────────┬────────────┘
                                           │
                                           ▼
                                  ┌─────────────────┐
                                  │                 │
                                  │ Response Agent  │
                                  │                 │
                                  └─────────────────┘
                                           │
                                           ▼
                                  ┌─────────────────┐
                                  │                 │
                                  │   Response to   │
                                  │      User       │
                                  │                 │
                                  └─────────────────┘
```

## Components

1. **Context Agent**: Extracts important information from user input and uses RAG to retrieve relevant knowledge.

2. **Sentiment Agent**: Analyzes the sentiment and intent of the user's message to determine if deflection is needed.

3. **Response Agent**: Generates the final response using the politician's fine-tuned model, incorporating context and sentiment information.

## Usage

### Command Line Interface

```bash
# Chat with Biden
python langgraph_politician.py cli chat --identity biden

# Chat with Trump with debug information
python langgraph_politician.py cli chat --identity trump --debug

# Process a single input and get JSON output
python langgraph_politician.py cli process --identity biden --input "What's your plan for healthcare?"

# Generate a visualization of the workflow
python langgraph_politician.py cli visualize
```

### API Server

```bash
# Start the API server
python langgraph_politician.py api
```

Then access the API documentation at http://localhost:8000/docs

## Technical Details

- **Models**: Uses fine-tuned Mistral 7B models with LoRA adapters for each politician identity
- **RAG**: Integrates with Milvus vector database to retrieve contextual information
- **LangGraph**: Orchestrates the workflow between specialized agents
- **Sentiment Analysis**: Analyzes input for hostility, bias, and "gotcha" questions to enable appropriate deflection strategies

## Extensions

The system is designed to be easily extended with additional:
- Politician identities (by adding new fine-tuned models)
- Agent capabilities (by adding new nodes to the graph)
- Response strategies (by modifying the workflow logic)

## Requirements

See `requirements/requirements-langgraph.txt` for the required packages. 