# Chat System Documentation

The AI Politician chat system provides an interactive interface for conversing with AI versions of politicians. The system is designed to simulate the speaking style, policy positions, and personality of specific politicians.

## Components

The chat system consists of several key files:

- `aipolitician.py`: The main entry point that provides a unified interface to all modes
- `scripts/chat_politician.py`: Script for clean chat experience
- `scripts/debug_politician.py`: Script for chat with debugging information
- `scripts/trace_politician.py`: Script for chat with detailed tracing
- `src/models/chat/chat_biden.py`: Biden-specific chat model implementation
- `src/models/chat/chat_trump.py`: Trump-specific chat model implementation

## Chat Modes

The system offers three main chat modes:

### 1. Clean Chat Mode

Provides a standard conversational experience without technical details.

```bash
python aipolitician.py chat biden
# or
python aipolitician.py chat trump
```

### 2. Debug Mode

Shows the clean chat response plus debugging information such as sentiment analysis, knowledge retrieval status, and deflection status.

```bash
python aipolitician.py debug biden
# or
python aipolitician.py debug trump
```

### 3. Trace Mode

Displays detailed information about each step in the workflow process, showing how the system analyzes the query and generates a response.

```bash
python aipolitician.py trace biden
# or
python aipolitician.py trace trump
```

## Optional Parameters

All modes support the following optional parameters:

- `--no-rag`: Disables the knowledge retrieval system (RAG), causing the politician to rely solely on the language model without additional factual knowledge

Example:
```bash
python aipolitician.py chat biden --no-rag
```

## Direct API Usage

For programmatic access, you can use the langgraph_politician.py script directly:

```bash
# Process a single input and get JSON output
python langgraph_politician.py process --identity biden --input "What's your economic policy?"
```

## Chat Implementation Details

The chat implementation for each politician is defined in separate files (`chat_biden.py` and `chat_trump.py`). These files include:

1. Specific prompting templates for each politician
2. Model configurations
3. Response generation logic
4. Knowledge integration methods

## Usage Examples

Here are some example queries to try with the system:

1. "What's your position on climate change?"
2. "How would you handle the crisis at the southern border?"
3. "What do you think about your political opponent?"
4. "Tell me about your economic policies"
5. "What's your vision for America's future?" 