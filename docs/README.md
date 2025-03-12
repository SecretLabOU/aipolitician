# AI Politician Project Structure

This document explains the organization of the AI Politician project.

## Project Structure

```
aipolitician/
├── src/                           # Main source code directory
│   ├── data/                      # Data handling components
│   │   ├── scraper/               # Web scraping functionality
│   │   ├── pipeline/              # Data processing pipeline
│   │   └── db/                    # Database functionality
│   ├── models/                    # Model training and inference
│   │   ├── training/              # Training scripts
│   │   └── chat/                  # Chat interface scripts
│   └── utils/                     # Shared utilities
├── tests/                         # All tests in one place
├── docs/                          # Documentation
├── requirements/                  # All requirements files
├── logs/                          # Centralized logs directory
└── setup.py                       # For making the package installable
```

## Component Descriptions

### src/data/scraper/
Contains scripts for scraping data from political sources, including speeches, tweets, and interviews.

### src/data/pipeline/
Data processing pipelines that transform raw scraped data into formats suitable for model training.

### src/data/db/
Database components, including Milvus vector database for RAG (Retrieval-Augmented Generation).

### src/models/training/
Scripts for training language models to mimic specific politicians' speaking styles.

### src/models/chat/
Interactive chat interfaces for interacting with the trained models.

### tests/
Unit and integration tests for different components of the system.

## Installation

```bash
# Install the base package
pip install -e .

# Install with specific components
pip install -e ".[scraper,training,chat]"
```

## Usage

See the main README.md file for detailed usage instructions. 