# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CONFIG_PATH=/root/.hybridrouter/config.json

# Set work directory
WORKDIR /app

# Install system dependencies (build-essential, curl, ca-certificates needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Pull local models during the build phase so they are baked into the image
# This ensures judges can run it out of the box with zero downloads.
RUN ollama serve > /var/log/ollama_build.log 2>&1 & \
    echo "Waiting for Ollama build daemon..." && \
    for i in {1..15}; do if curl -s http://127.0.0.1:11434/api/tags >/dev/null; then break; fi; sleep 1; done && \
    echo "Pulling qwen2.5:0.5b..." && ollama pull qwen2.5:0.5b && \
    echo "Pulling smollm2:135m..." && ollama pull smollm2:135m && \
    echo "Pulling smollm2:360m..." && ollama pull smollm2:360m

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create the config directory and copy the pre-calibrated configuration preset
RUN mkdir -p /root/.hybridrouter && \
    cp /app/calibration/config_preset.json /root/.hybridrouter/config.json

# Make entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Use entrypoint.sh to start Ollama and run cli/main.py
ENTRYPOINT ["/app/entrypoint.sh"]
