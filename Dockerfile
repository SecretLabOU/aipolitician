# Base image using NVIDIA CUDA 12.x with Ubuntu
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/home/appuser/.local/bin:${PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    python3-dev \
    build-essential \
    wget \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -s /bin/bash appuser

# Create directories for data and models
RUN mkdir -p /app/models /app/data \
    && chown -R appuser:appuser /app

# Set working directory
WORKDIR /app

# Copy requirements files
COPY requirements-chat.txt requirements-training.txt ./

# Switch to non-root user
USER appuser

# Create and activate virtual environment
RUN python3.10 -m venv /home/appuser/venv
ENV PATH="/home/appuser/venv/bin:$PATH"

# Set environment variables for CUDA
ENV NVIDIA_VISIBLE_DEVICES=all
ENV CUDA_VISIBLE_DEVICES=0

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-chat.txt \
    && pip install --no-cache-dir -r requirements-training.txt \
    # Install PyTorch with CUDA support
    && pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy application code
COPY --chown=appuser:appuser . /app

# Set environment variables for the application
ENV PYTHONPATH=/app:$PYTHONPATH
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models
ENV HF_DATASETS_CACHE=/app/data

# Default command to run the application
ENTRYPOINT ["python", "chat_biden.py"]

