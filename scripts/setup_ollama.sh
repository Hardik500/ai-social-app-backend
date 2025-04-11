#!/bin/bash
# Pull the required Ollama model for embeddings

echo "Setting up Ollama models..."

# Wait for Ollama service to be ready
echo "Waiting for Ollama service..."
for i in {1..30}; do
  if curl -s http://ollama:11434/api/health >/dev/null; then
    echo "Ollama service is ready!"
    break
  fi
  echo "Waiting for Ollama service... ($i/30)"
  sleep 2
  if [ $i -eq 30 ]; then
    echo "Error: Ollama service didn't start in time"
    exit 1
  fi
done

# Pull the embedding model
echo "Pulling nomic-embed-text model..."
ollama pull nomic-embed-text

echo "Ollama setup complete!" 