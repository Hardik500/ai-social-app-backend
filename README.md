# AI Social App Backend

A backend application for a personality-based social app that allows uploading Slack conversations and associating them with personalities.

## Features

- Upload Slack conversations via API endpoint
- Store conversations with embeddings in PostgreSQL with TimescaleDB and pgAI
- Associate conversations with specific users/personalities
- Asynchronous embedding generation using pgAI's vectorizer-worker
- Duplicate conversation detection to prevent data duplication

## Setup

### Prerequisites

- Docker and Docker Compose
- Ollama for embeddings

### Running the Application

1. Clone the repository
2. Navigate to the project directory
3. Start the application with Docker Compose:

```bash
docker-compose up -d
```

The API will be available at http://localhost:8000

## API Usage

### Upload a Conversation

```
POST /conversations/
```

Request body example:

```json
{
  "source": "slack",
  "messages": [
    {
      "user": "Hardik",
      "timestamp": "1716292517.637399",
      "message": "Got it, then lets have it offline next week only"
    },
    {
      "user": "Murali",
      "timestamp": "1716292510.807159",
      "message": "awesome. will catch up with you in person then"
    },
    {
      "user": "Murali",
      "timestamp": "1716292490.830249",
      "message": "so most of next week and BRP week"
    }
  ],
  "user_info": {
    "username": "Hardik",
    "email": "hardik@example.com",
    "phone": "1234567890",
    "description": "Tech enthusiast and developer"
  }
}
```

The endpoint will automatically detect and prevent duplicate conversations based on the source and message timestamps. If a duplicate is detected, the API will return the existing conversation with a 200 status code and include duplicate detection information.

To bypass duplicate detection (if needed), you can use the `skip_duplicate_check` query parameter:

```
POST /conversations/?skip_duplicate_check=true
```

### Get All Conversations

```
GET /conversations/
```

### Get a Specific Conversation

```
GET /conversations/{conversation_id}
```

### Verify Message Embedding

```
GET /conversations/messages/{message_id}/embedding
```

### Search for Similar Messages

```
GET /conversations/search/similar?query=Let's meet next week
```

### Reset Database (Admin Endpoint)

```
POST /admin/reset-database
```

> ⚠️ **WARNING**: This endpoint will drop all tables and recreate them, resulting in loss of all data. Use with caution!

## Testing and Verification

### Running Tests

The application includes a comprehensive test suite to verify functionality. To run tests:

```bash
# Install test dependencies
pip install -e ".[test]"

# Run the test suite
pytest
```

### Sanity Check

A sanity check script is provided to verify the end-to-end flow of the application. This script:

1. Checks API health
2. Uploads a sample conversation
3. Retrieves the conversation
4. Verifies embedding generation
5. Performs a similarity search
6. Tests duplicate detection
7. Tests non-duplicate detection

To run the sanity check:

```bash
# With the application running
python scripts/sanity_check.py
```

This provides a quick way to verify that all components are working correctly after setup or changes.

## Data Persistence

The application uses named Docker volumes to ensure data is persisted across restarts:

- `postgres_data_ai_social_app` - Stores PostgreSQL database files
- `ollama_data_ai_social_app` - Stores Ollama models and data

This ensures that your data remains available even if you restart the containers or your system.

## Architecture

The application uses the following components:

- **FastAPI** - Web framework for the API
- **TimescaleDB** - PostgreSQL database with time-series and vector extensions
- **pgAI** - PostgreSQL extension for AI capabilities and embeddings
- **Ollama** - Local AI model provider for generating embeddings
- **pgAI Vectorizer Worker** - Asynchronous worker for generating embeddings

When a conversation is uploaded:
1. The system checks for potential duplicates based on source and timestamp patterns
2. If not a duplicate, the conversation and its messages are stored in the database
3. The vectorizer worker is notified to generate embeddings for each message
4. Embeddings are stored in the database for later search and retrieval

## Development

To run the application in development mode:

```bash
docker-compose up
```

This will start the application with hot-reloading enabled.

## Environment Variables

The application uses the following environment variables:

- `DATABASE_URL`: PostgreSQL connection string
- `OLLAMA_BASE_URL`: URL for Ollama API
- `API_HOST`: Host to bind the API server
- `API_PORT`: Port to bind the API server
- `ENVIRONMENT`: Application environment (development, production) 