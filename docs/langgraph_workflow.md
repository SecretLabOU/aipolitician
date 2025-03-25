# LangGraph Workflow Documentation

The AI Politician system uses LangGraph to create a structured workflow for processing user inputs and generating responses. This documentation explains the LangGraph implementation and how it orchestrates different components of the system.

## Overview

LangGraph is used to create a directed graph that processes user inputs through a series of agents. Each agent performs a specific task in the overall workflow:

1. **Context Agent**: Extracts context and retrieves relevant knowledge
2. **Sentiment Agent**: Analyzes the sentiment and intent of the user's input
3. **Response Agent**: Generates a politician-specific response

## Workflow Structure

The workflow is defined in `src/models/langgraph/workflow.py` and follows this sequence:

```
[Entry] -> Context Agent -> Sentiment Agent -> Response Agent -> [End]
```

## Key Components

### 1. Input/Output Schemas

The workflow uses Pydantic models to define the expected input and output:

- `PoliticianInput`: Contains user input, politician identity, RAG setting, and trace flag
- `PoliticianOutput`: Contains the generated response, sentiment analysis, deflection status, and knowledge retrieval status

### 2. State Management

The workflow maintains state using a TypedDict called `WorkflowState` which includes:

- `user_input`: The user's query or statement
- `politician_identity`: Which politician to impersonate
- `use_rag`: Whether to use knowledge retrieval
- `trace`: Whether to display trace information
- `context`: Retrieved context and knowledge
- `has_knowledge`: Whether relevant knowledge was found
- `sentiment_analysis`: Results of sentiment analysis
- `should_deflect`: Whether to use deflection strategies
- `response`: The final generated response

### 3. Agent Functions

Each agent is implemented as a function that takes and returns a state dictionary:

- `trace_context_agent()`: Wraps the context extraction agent with tracing capabilities
- `trace_sentiment_agent()`: Wraps the sentiment analysis agent with tracing capabilities
- `trace_response_agent()`: Wraps the response generation agent with tracing capabilities

### 4. Main Workflow

The function `create_politician_graph()` creates the LangGraph structure, defining nodes and edges to create the workflow.

## Using the Workflow

The `process_user_input()` function is the main entry point for using the workflow:

```python
from src.models.langgraph.workflow import process_user_input, PoliticianInput

# Create input
input_data = PoliticianInput(
    user_input="What's your position on climate change?",
    politician_identity="biden",
    use_rag=True,
    trace=False
)

# Process through workflow
result = process_user_input(input_data)

# Access result
print(result.response)
```

## Visualization

You can visualize the workflow graph using:

```bash
python langgraph_politician.py visualize
```

This creates an HTML visualization of the workflow graph structure.

## Extending the Workflow

To add new capabilities:

1. Update the `WorkflowState` TypedDict with new state fields
2. Create new agent functions that process and update the state
3. Add new nodes to the graph in `create_politician_graph()`
4. Update the graph edges to incorporate the new nodes into the workflow 