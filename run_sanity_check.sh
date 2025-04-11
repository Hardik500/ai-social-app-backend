#!/bin/bash
# Run sanity check script inside the API container

echo "Running sanity check for AI Social App Backend..."
docker-compose exec api python scripts/sanity_check.py 