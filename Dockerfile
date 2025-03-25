# Base image using lightweight NVIDIA CUDA
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH=/app:$PYTHONPATH \
    HF_HOME=/app/models \
    TRANSFORMERS_CACHE=/app/models

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
RUN useradd -m -s /bin/bash appuser \
    && mkdir -p /app/models \
    && chown -R appuser:appuser /app

# Set up Python environment
FROM base AS python-deps

WORKDIR /app

# Copy only requirements file first for better layer caching
COPY --chown=appuser:appuser requirements.txt .

# Switch to non-root user
USER appuser

# Create and activate virtual environment
RUN python3.10 -m venv /home/appuser/venv
ENV PATH="/home/appuser/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python-deps

# Copy application code
COPY --chown=appuser:appuser . /app

# Set default command
ENTRYPOINT ["python", "chat.py"]

# Set default command-line arguments (can be overridden)
CMD ["--persona", "trump"]

