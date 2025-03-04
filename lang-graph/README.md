# Political AI

Talk to AI-powered politicians - simulating their views, speech patterns, and policy positions.

## Quick Start

```bash
# Setup the models first
./setup_models.sh

# Simple - talk to a politician directly
python src/main.py donald_trump

# See who's available 
python src/main.py --list

# Try a quick demo
python src/main.py --demo
```

## Features

- 🗣️ **Talk to famous politicians** like Trump, AOC, Sanders, and Ardern
- 🧠 **Ask anything** about their policies, views, or current events
- 🔄 **Compare perspectives** on topics like healthcare, economy, or climate
- 🎭 **Natural conversations** with realistic speech patterns
- 🤖 **Fine-tuned Trump model** for authentic responses

## Available Politicians

- **Donald Trump** - Republican (use: `donald_trump`)
- **Alexandria Ocasio-Cortez** - Democrat (use: `alexandria_ocasio_cortez`)
- **Bernie Sanders** - Independent (use: `bernie_sanders`)
- **Jacinda Ardern** - New Zealand Labour (use: `jacinda_ardern`)

## Example Usage

```
# Talk to Bernie Sanders
$ python src/main.py bernie_sanders

Now talking with: Bernie Sanders (Independent)

Ask any question or type 'exit' to quit

> What do you think about healthcare?

Bernie Sanders: Healthcare is a human right, not a privilege. In the United States, 
we are the only major country that doesn't guarantee healthcare to all people. 
It is unconscionable that millions of Americans cannot afford to see a doctor when 
they get sick. We need Medicare for All, which would provide comprehensive healthcare 
to every man, woman, and child in this country while saving the average family 
thousands of dollars per year.

> exit

Goodbye!
```

## Python API

Want to use this in your own project? It's simple:

```python
import asyncio
from political_agent_graph import run_conversation, select_persona

async def main():
    # Choose a politician
    select_persona("donald_trump")
    
    # Ask a question
    response = await run_conversation("What's your view on immigration?")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
```

## Adding New Politicians

Want to add your own politician? Just create a new entry in `src/political_agent_graph/personas.json` with:

```json
{
  "id": "politician_id",
  "name": "Politician Name",
  "role": "Current role",
  "party": "Political party",
  "biography": { ... },
  "speech_patterns": { ... },
  "policy_stances": { ... }
}
```

## License

MIT License