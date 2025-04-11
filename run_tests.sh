#!/bin/bash
# Run tests inside the API container

echo "Running tests for AI Social App Backend..."
docker-compose exec api pytest -v 