# AI Politician Debate System

A LangGraph-based workflow system that enables different AI politicians to debate each other in a structured format.

## Features

- **Multi-Party Debates**: Support for two or more AI politicians debating simultaneously
- **Configurable Formats**: Multiple debate formats including town hall, head-to-head, and panel discussions
- **Turn-Taking & Topic Management**: Structured turn-taking with automatic topic progression
- **Fact-Checking**: Automatic fact-checking of politician statements
- **Interruptions & Rebuttals**: Support for dynamic interruptions and targeted rebuttals
- **Moderation**: Configurable moderator control levels

## Usage

### Running a Debate

```bash
# Basic head-to-head debate
python scripts/run_debate.py run --topic "Climate Change" --participants "biden,trump"

# Town hall format with multiple participants
python scripts/run_debate.py run --topic "Healthcare Reform" --participants "biden,trump,sanders" --format "town_hall"

# Enable interruptions and fact-checking
python scripts/run_debate.py run --topic "Economy" --participants "biden,trump" --allow-interruptions --fact-check

# Save debate transcript to file
python scripts/run_debate.py run --topic "Immigration" --participants "biden,trump" --output "debate_transcript.json"

# Disable RAG knowledge retrieval
python scripts/run_debate.py run --topic "Foreign Policy" --participants "biden,trump" --no-rag

# Show trace information
python scripts/run_debate.py run --topic "Education" --participants "biden,trump" --trace
```

### Command-Line Options

```
usage: run_debate.py run [-h] --topic TOPIC --participants PARTICIPANTS
                        [--format {town_hall,head_to_head,panel}]
                        [--time-per-turn TIME_PER_TURN] [--allow-interruptions]
                        [--fact-check] [--moderator-control {strict,moderate,minimal}]
                        [--no-rag] [--trace] [--output OUTPUT]

Run a debate between AI politicians with the following options:

  --topic TOPIC         Main debate topic
  --participants PARTICIPANTS
                        Comma-separated list of politician identities (e.g., 'biden,trump')
  --format {town_hall,head_to_head,panel}
                        Debate format (default: head_to_head)
  --time-per-turn TIME_PER_TURN
                        Time in seconds allocated per turn (default: 60)
  --allow-interruptions
                        Allow interruptions during the debate
  --fact-check          Enable fact checking (default: enabled)
  --moderator-control {strict,moderate,minimal}
                        Level of moderator control (default: moderate)
  --no-rag              Disable RAG knowledge retrieval
  --trace               Show trace information during the debate
  --output OUTPUT       Output file for debate transcript (JSON)
```

### Visualizing the Workflow

```bash
# Visualize the debate workflow graph
python scripts/run_debate.py visualize
```

### Configuration

```bash
# List available politician identities
python scripts/run_debate.py config --list-politicians

# List available debate formats
python scripts/run_debate.py config --list-formats
```

## Debate Formats

- **Head-to-Head**: Direct debate between two politicians with alternating turns
- **Town Hall**: Multiple politicians answering questions, often with a focus on audience concerns
- **Panel**: Multiple politicians with a moderator leading discussion, similar to a roundtable

## How It Works

The AI Politician Debate System uses a LangGraph workflow to connect multiple AI agents:

1. **Moderator Agent**: Controls turn-taking, introduces topics, and maintains debate structure
2. **Politician Agents**: Generate responses based on their identity, knowledge, and the debate context
3. **Fact-Checking Agent**: Verifies factual claims made by politicians during the debate
4. **Topic Manager**: Handles progression through subtopics to ensure comprehensive coverage
5. **Interruption Handler**: Manages interruptions between politicians for more dynamic debates

The system implements a flexible state machine that allows for different debate formats and configurations.

## Requirements

- Python 3.8+
- LangGraph
- Dependencies from the AI Politician project

## Examples

### Example Debate Transcript

```
================================================================================
DEBATE: Climate Change
PARTICIPANTS: biden, trump
================================================================================

MODERATOR: Welcome to today's head_to_head debate on the topic of 'Climate Change'. Participating in this debate are biden, trump. Each speaker will have 60 seconds per turn. Interruptions will be allowed during this debate. Statements will be fact-checked for accuracy. Let's begin with biden.

BIDEN: Climate change is an existential threat that requires immediate action. Under my administration, we've made historic investments in clean energy through the Inflation Reduction Act, committing over $360 billion to address climate change. We're on track to cut emissions in half by 2030 and reach net-zero by 2050. We've rejoined the Paris Climate Agreement and are working with our international partners to hold all nations accountable. This isn't just about saving our planet—it's about creating good-paying union jobs in clean energy, revitalizing American manufacturing, and leading the global economy of the future.

[FACT CHECK] Statements by BIDEN:
  • Claim: "We're on track to cut emissions in half by 2030"
    Rating: PARTIALLY TRUE (0.75)
    Sources: Congressional Budget Office report (2022), PolitiFact fact check (2023)

  • Claim: "We've made historic investments in clean energy through the Inflation Reduction Act, committing over $360 billion to address climate change"
    Rating: MOSTLY TRUE (0.89)
    Sources: Department of Health study (2021), Wall Street Journal investigation (2023)

MODERATOR: Your time is up. trump, your response?

TRUMP: Biden's climate agenda is killing American jobs and crushing our economy. These radical Green New Deal policies are sending energy prices through the roof, while China and India continue to build coal plants every week. When I was President, we had energy independence for the first time, with lower gas prices and more American energy jobs. Climate change has been happening for millions of years, but these terrible Paris Accord restrictions only hurt American workers while letting other countries pollute with zero consequences. We need clean air and crystal clean water, but we don't need to destroy our economy to get it.

[INTERRUPTION] BIDEN: That's simply not true! trump is misleading the audience about climate change. The clean energy transition is creating millions of new jobs, and our economy has been growing while em

MODERATOR: Let's move on to discuss renewable energy.

BIDEN: Renewable energy is the future, and America should lead it. During my administration, we've seen record growth in solar and wind deployment. The cost of renewable energy has plummeted, making it cheaper than fossil fuels in many parts of the country. We're making historic investments in upgrading our power grid and building electric vehicle charging stations nationwide. This transition will create millions of good-paying jobs that can't be outsourced. Republicans like my opponent want to cling to the energy sources of the past, but I believe America must innovate and lead the global clean energy economy.

[FACT CHECK] Statements by BIDEN:
  • Claim: "The cost of renewable energy has plummeted, making it cheaper than fossil fuels in many parts of the country"
    Rating: MOSTLY TRUE (0.87)
    Sources: Department of Health study (2021), New York Times analysis (2022), Wall Street Journal investigation (2023)

MODERATOR: Your time is up. trump, your response?

TRUMP: Look, I'm all for renewable energy. I'm for solar, I'm for wind, I'm for everything. But it has to be affordable and it can't collapse our energy grid. These windmills kill all the birds and they make a terrible noise. And when the wind doesn't blow, you don't have power. That's a big problem. Under Biden, electricity prices are up 30% and gas prices through the roof. We need an all-of-the-above approach - oil, gas, coal, nuclear, and renewables. That's how you have true energy dominance. America sits on more energy reserves than any other nation. We should be using it!

[FACT CHECK] Statements by TRUMP:
  • Claim: "Under Biden, electricity prices are up 30%"
    Rating: PARTIALLY FALSE (0.45)
    Correction: The actual facts differ: Under Biden, electricity prices are down 30%
    Sources: Bureau of Labor Statistics data (2023)

================================================================================
DEBATE SUMMARY:
  Topic: Climate Change
  Participants: biden, trump
  Turns: 4
  Fact Checks: 3
  Subtopics Covered: Climate Change, renewable energy
================================================================================
```

## Extending the System

The debate system can be extended in several ways:

1. Add new politician identities in `src/models/langgraph/config.py`
2. Implement custom debate formats by extending the `DebateFormat` class
3. Enhance fact-checking capability by connecting to external verification systems
4. Add support for audience questions in town hall format
5. Implement real-time visualization of debate dynamics and sentiment analysis 