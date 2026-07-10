#!/bin/bash
set -e

# Start Ollama server in background
echo "Starting Ollama server..."
ollama serve > /var/log/ollama.log 2>&1 &

# Wait for Ollama server to respond on 11434
echo "Waiting for Ollama to wake up..."
for i in {1..20}; do
  if curl -s http://127.0.0.1:11434/api/tags >/dev/null; then
    echo "Ollama is ready."
    break
  fi
  sleep 1
done

# Run the CLI tool with arguments passed to the container
exec python cli/main.py "$@"
