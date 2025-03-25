# Political Agent LangGraph

A streamlined, GPU-optimized conversation system for political personas using LangGraph and LLaMA-cpp.

## Overview

This system provides an efficient implementation for running conversations with political personas. It uses:

- **GPU-Accelerated LLMs**: Optimized for RTX 4090 and RTX 4060 Ti
- **Parallel Processing**: Faster response generation
- **LangGraph**: Structured conversation flow
- **RAG Integration**: Fact-based responses (optional)

## Quick Start

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the demo:
   ```
   python src/main.py --demo
   ```

3. Start a conversation:
   ```
   python src/main.py --chat trump
   ```

## Setup

### Local Models

Place your GGUF model files in the `models` directory and configure in `models/config.json`:

```json
{
  "trump": {
    "model_path": "models/trump-7b.gguf"
  },
  "biden": {
    "model_path": "models/biden-7b.gguf"
  }
}
```

The system will auto-detect your GPU configuration and optimize accordingly.

### Usage

```
python src/main.py --help

options:
  -h, --help            show this help message and exit
  --chat PERSONA, -c PERSONA
                        Start chat with persona (e.g., trump, biden)
  --demo, -d            Run demonstration
  --list, -l            List available personas
```

## Architecture

The system uses a streamlined LangGraph implementation:

1. **Initial Processing**: Analyzes sentiment, determines topic, and retrieves context
2. **Strategy**: Decides whether to deflect
3. **Response Generation**: Creates policy stance and formats response
4. **Fact Checking**: Verifies responses against retrieved information

All components are optimized for performance and efficiency.

## GPU Support

The system automatically:
- Detects available CUDA devices
- Optimizes context size based on available VRAM
- Uses optimal parallelization for your hardware

## Docker Support

For containerized deployment:

```
docker-compose up
```

OR

```
./run.sh
```

## Features

- 🗣️ **Talk to politicians** like Trump, AOC, Sanders, and Ardern
- 🧠 **Authentic responses** on policies, views, and current events
- 🔄 **Compare perspectives** on topics across the political spectrum
- 🤖 **Fine-tuned Trump model** for highly realistic language patterns

## Available Politicians

- **Donald Trump** - Republican (use: `donald_trump`)

*More politicians coming soon!*

## Example Usage

```
# Talk to Donald Trump
$ python src/main.py donald_trump

Talking with: Donald Trump (Republican)
Ask any question or type 'exit' to quit

> What's your view on immigration?

Donald Trump: We have a disaster at our border, a total disaster. Millions of people pouring in, and nobody even knows where they're coming from. It's a disgrace, a total disgrace. When I was president - and I will be again - we had the strongest border in the history of our country. The wall was being built, and it was working beautifully.

Look, I'm all for legal immigration. Legal immigrants, they love our country. But these people are coming in illegally, bringing crime, bringing drugs - the cartels are making billions. It's destroying our country from within.

We need to finish the wall and implement strong border policies. Other countries are laughing at us right now. They're emptying their prisons and mental institutions, sending them straight to America. It's going to stop, believe me.

> exit

Goodbye!
```

## License

MIT License