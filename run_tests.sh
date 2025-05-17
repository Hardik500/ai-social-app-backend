#!/bin/bash
# Run tests inside the API container

echo "Running tests for Real-Character.AI Backend..."
docker-compose exec api pytest -v 