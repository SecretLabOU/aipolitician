# Political AI

Talk to AI-powered politicians that simulate real-world rhetorical styles and policy positions.

## Quick Start

```bash
# Setup required models
./setup_models.sh

# Talk to a politician directly
python src/main.py donald_trump

# See available politicians
python src/main.py --list

# Try a quick demo
python src/main.py --demo
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