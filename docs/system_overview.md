# AI Politician System Overview

The AI Politician is a system designed to simulate political figures, providing responses that mimic their speaking style, policy positions, and personality. The system uses a combination of language models, knowledge retrieval, and sentiment analysis to generate authentic-sounding responses to user queries.

## System Architecture

The AI Politician system is built with the following key components:

1. **Chat Interface**: The main user interaction layer that allows conversational interaction with the AI politician.

2. **LangGraph Workflow**: A graph-based workflow that orchestrates the process of generating responses, built using LangGraph.

3. **Knowledge Database**: A vector database (Milvus) that stores political knowledge, speeches, policy positions, and other relevant information.

4. **Scraper**: A tool for collecting data about politicians from reliable sources.

5. **Data Pipeline**: Processes scraped data into a format suitable for the knowledge database.

6. **Training System**: Fine-tunes language models to capture the speaking style and policy positions of specific politicians.

## System Flow

1. A user enters a question or statement to the AI politician.
2. The system analyzes the query to understand its intent and sentiment.
3. Relevant knowledge is retrieved from the database if available.
4. The system determines if the query is hostile or requires deflection.
5. A response is generated based on the identity of the politician, the context, and whether deflection is needed.
6. The response is presented to the user, along with optional debug information.

## Supported Politicians

Currently, the system supports the following politicians:
- Joe Biden
- Donald Trump

## Usage Modes

The system supports several operating modes:
- **Chat Mode**: Clean conversational interface
- **Debug Mode**: Shows additional information about analysis and decision-making
- **Trace Mode**: Displays detailed workflow execution information

See the individual component documentation for more details on each part of the system. 