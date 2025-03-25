# Installation Guide

This document provides detailed instructions for installing and setting up the AI Politician system.

## System Requirements

- **Operating System**: Linux, macOS, or Windows
- **Python**: 3.8+ (recommended: Python 3.10)
- **RAM**: 8GB minimum (16GB+ recommended)
- **Storage**: 5GB for base system, 20GB+ with models and database
- **GPU**: NVIDIA GPU with 8GB+ VRAM recommended for model training
- **CUDA**: 12.0+ for GPU acceleration
- **Docker**: Required for Milvus database

## Installation Options

You can install the AI Politician system in several ways depending on your needs:

### Option 1: Full Installation

Install all components for a complete system:

1. Clone the repository
2. Set up a virtual environment
3. Install all requirements
4. Set up the Milvus database

### Option 2: Component-based Installation

Install only the components you need:

- **Chat System**: For interacting with AI Politicians
- **LangGraph**: For the main workflow without scraper or training
- **Scraper**: For collecting data
- **Training**: For training models

## Detailed Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/aipolitician.git
cd aipolitician
```

### 2. Create a Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate on Linux/macOS
source .venv/bin/activate

# Activate on Windows
.venv\Scripts\activate
```

### 3. Install Requirements

#### Option 1: Install All Requirements

```bash
pip install -r requirements/requirements-all.txt
```

#### Option 2: Install Individual Components

**Base Requirements** (always needed):
```bash
pip install -r requirements/requirements-base.txt
```

**Chat System**:
```bash
pip install -r requirements/requirements-chat.txt
```

**LangGraph Workflow**:
```bash
pip install -r requirements/requirements-langgraph.txt
```

**Scraper**:
```bash
pip install -r requirements/requirements-scraper.txt
```

**Training**:
```bash
pip install -r requirements/requirements-training.txt
```

### 4. Set Up Milvus Database

#### Option 1: Using Docker

```bash
# Start a local Milvus instance
docker run -d --name milvus_standalone -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest standalone
```

#### Option 2: Using Docker Compose

Create a `docker-compose.yml` file:

```yaml
version: '3.5'

services:
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.0
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/etcd:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd

  minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2020-12-03T00-03-10Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/minio:/minio_data
    command: minio server /minio_data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  standalone:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.2.2
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/milvus:/var/lib/milvus
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - "etcd"
      - "minio"
```

Then run:

```bash
docker-compose up -d
```

### 5. Load Data into Milvus

To load processed data into the Milvus database:

```bash
python scripts/load_milvus_data.py --path ./data/processed/biden --politician biden
python scripts/load_milvus_data.py --path ./data/processed/trump --politician trump
```

## Verification

To verify your installation is working correctly:

```bash
# Verify system dependencies
python check_system.py

# Start a chat session to test
python aipolitician.py chat biden
```

## Development Setup

For development, you may want to install in editable mode:

```bash
pip install -r requirements/requirements-base.txt
pip install -r requirements/requirements-langgraph.txt
```

## Troubleshooting

### Common Issues

#### Milvus Connection Issues

If you encounter errors connecting to Milvus:

1. Ensure Docker is running
2. Check container status: `docker ps | grep milvus`
3. Check logs: `docker logs milvus-standalone`

#### Model Loading Issues

If you have problems loading models:

1. Check your internet connection (models may need to be downloaded)
2. Ensure you have access to the Hugging Face models
3. Verify GPU/CUDA setup if using GPU acceleration

#### Import Errors

If you see Python import errors:

1. Ensure your virtual environment is activated
2. Verify all requirements are installed
3. Check for version conflicts between packages 