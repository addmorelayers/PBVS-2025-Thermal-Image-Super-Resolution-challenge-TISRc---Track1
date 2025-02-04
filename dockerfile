# Use Python base image
FROM --platform=linux/arm64 python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    torch \
    pytorch-lightning \
    numpy \
    matplotlib \
    torchvision \
    tqdm \
    torchmetrics \
    PyYAML

# Set working directory
WORKDIR /app
