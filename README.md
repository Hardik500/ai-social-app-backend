# AI Social App Backend

A backend application for a personality-based social app that allows uploading Slack conversations and associating them with personalities.

## Features

- Upload Slack conversations via API endpoint
- Store conversations with embeddings in PostgreSQL with TimescaleDB and pgAI
- Associate conversations with specific users/personalities
- Asynchronous embedding generation using pgAI's vectorizer-worker
- Automatic embedding updates when content changes
- Retrieval Augmented Generation (RAG) for answering questions from conversation data
- Duplicate conversation detection to prevent data duplication
- Caching system for fast response generation
- Parallel processing for performance optimization
- Streaming responses for better user experience

## Performance Optimizations

### Caching System
- In-memory caching for frequently accessed responses
- Redis-based persistent caching (configurable via `REDIS_URL`)
- Automatic cache invalidation and pruning

### Parallel Processing
- Batch processing of messages for analysis
- Concurrent embedding generation
- Background preloading of common questions

### Streaming Responses
- Real-time token streaming for faster perceived response times
- Compatible with front-end incremental display

### Other Optimizations
- Rate limiting for external API calls
- Background tasks for non-critical operations
- Preloading of related follow-up questions

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

### Data Ingestion

The application supports ingesting data from multiple sources to create conversations:

#### Upload Source Data

```
POST /ingestion/
```

This endpoint accepts multipart/form-data with the following fields:
- `source_type`: Type of the source (e.g., "slack_har", "whatsapp")
- `source_file`: File containing the source data (e.g., HAR file for Slack)
- `primary_user_info`: JSON string with primary user information
- `additional_users`: Optional JSON string with additional users information
- `user_mapping`: Optional JSON string with mapping from source user IDs to usernames

Example primary_user_info:
```json
{
  "username": "Hardik",
  "email": "hkhandelwal@example.com",
  "phone": "+919680023483",
  "description": "Sr. Software Engineer"
}
```

Example additional_users:
```json
[
  {
    "username": "Murali",
    "email": "murali.v@example.com",
    "phone": "+971543406899",
    "description": "Head of Engineering, Clari Copilot"
  }
]
```

Example user_mapping (for Slack):
```json
{
  "U055WM6DTJL": "Hardik",
  "U03HPSXQXHC": "Murali"
}
```

The `user_mapping` parameter is especially important for Slack HAR files, as it maps the cryptic Slack user IDs (like "U055WM6DTJL") to human-readable usernames (like "Hardik"). Without this mapping, the system will create users with usernames like "unknown_U055WM6DTJL".

The endpoint extracts messages from the source file and stores them in the database, associating them with the specified users.

#### Test Ingestion (JSON Data)

```
POST /ingestion/test
```

This endpoint is for testing and accepts JSON data directly instead of file upload:

```json
{
  "source_type": "slack_har",
  "source_data": { /* HAR file data as JSON */ },
  "primary_user_info": {
    "username": "Hardik",
    "email": "hkhandelwal@example.com",
    "phone": "+919680023483",
    "description": "Sr. Software Engineer"
  },
  "additional_users": [
    {
      "username": "Murali",
      "email": "murali.v@example.com",
      "phone": "+971543406899",
      "description": "Head of Engineering, Clari Copilot"
    }
  ],
  "user_mapping": {
    "U055WM6DTJL": "Hardik",
    "U03HPSXQXHC": "Murali"
  }
}
```

### Supported Data Sources

Currently, the application supports the following data sources:

1. **Slack HAR Files**: HTTP Archive (HAR) files exported from Slack web interface
2. **WhatsApp**: (*Coming soon*) - Export files from WhatsApp

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
  "primary_user_info": {
    "username": "Hardik",
    "email": "hardik@example.com",
    "phone": "1234567890",
    "description": "Tech enthusiast and developer"
  },
  "additional_users": [
    {
      "username": "Murali",
      "email": "murali@example.com",
      "phone": "9876543210",
      "description": "Product manager"
    }
  ]
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

### Retrieval Augmented Generation (RAG)

```
POST /conversations/rag?query=What is the project deadline?&max_context_messages=3
```

The RAG endpoint combines semantic search with generative AI to provide contextually accurate answers:

1. It retrieves the most semantically similar messages based on vector embeddings
2. Uses these messages as context for the LLM to generate a response
3. Returns both the generated answer and the context that was used

Response example:

```json
{
  "query": "What is the project deadline?",
  "answer": "Based on the provided context, the project deadline is March 15th, 2023.",
  "context_used": [
    "Message from john_doe: We need to finish the project by March 15th, 2023. (Similarity: 0.89)",
    "Message from project_manager: Don't forget about the deadline next month! (Similarity: 0.76)",
    "Message from team_lead: The client expects all deliverables by mid-March. (Similarity: 0.71)"
  ],
  "model_used": "llama3"
}
```

You can optionally specify a different model to use with the `model` parameter:

```
POST /conversations/rag?query=What is the roadmap?&model=gemma
```

### Personality Profile Management

#### Generate a Personality Profile

```
POST /personalities/users/{username}/generate
```

This endpoint analyzes all messages from a user to generate a detailed personality profile. The profile includes:

- Personality traits (Big Five dimensions)
- Communication style
- Interests and values
- A system prompt for simulating user responses

A user must have at least 5 messages in the database to generate a profile.

#### Get User's Personality Profiles

```
GET /personalities/users/{username}
```

You can add the `active_only=true` query parameter to get only the currently active profile:

```
GET /personalities/users/{username}?active_only=true
```

#### Get a Specific Profile by ID

```
GET /personalities/profiles/{profile_id}
```

#### Ask a Question to a User (Personality Simulation)

```
POST /personalities/users/{username}/ask
```

Request body example:

```json
{
  "question": "What do you think about using React for this project?"
}
```

Response example:

```json
{
  "question": "What do you think about using React for this project?",
  "answer": "I think React would be a solid choice for the frontend. I've used it on several projects and it's quite efficient for building component-based UIs. The ecosystem is mature and there's plenty of support available when you run into issues.",
  "username": "Hardik"
}
```

This endpoint uses the user's personality profile to generate a response that matches their communication style, interests, and personality traits.

#### Enhanced Personality Response with RAG

```
POST /personalities/users/{username}/ask/rag
```

This version uses Retrieval Augmented Generation to provide more accurate and evidence-based personality responses. It finds actual messages the user has sent that are relevant to the question, then uses them to enhance the response.

Request body is the same as the regular ask endpoint:

```json
{
  "question": "What do you think about using React for this project?"
}
```

But the response includes the relevant messages that were used as context:

```json
{
  "question": "What do you think about using React for this project?",
  "answer": "I think React would be a solid choice for the frontend. I have previously worked with React and found it quite efficient for component-based UIs. As I mentioned before, its ecosystem is mature and there's plenty of support available.",
  "username": "Hardik",
  "used_messages": 3,
  "relevant_context": [
    "I've been using React for several projects and it's working great for us",
    "The React component model makes it easy to reuse UI elements across the application",
    "I prefer frontend frameworks with good community support and documentation"
  ]
}
```

You can control the number of similar messages to include as context:

```
POST /personalities/users/{username}/ask/rag?max_context_messages=3
```

##### Streaming Responses

For improved user experience, you can request streaming responses by adding the `stream=true` query parameter:

```
POST /personalities/users/{username}/ask?stream=true
```

The response will be a stream of JSON objects, each containing a token or chunk of the response:

```json
{"type":"token","content":"I ","done":false}
{"type":"token","content":"think ","done":false}
{"type":"token","content":"React ","done":false}
...
{"type":"complete","content":"I think React would be a solid choice for the frontend...","done":true}
```

For cached responses, you'll receive a single chunk with the complete answer:

```json
{"type":"cached","content":"I think React would be a solid choice for the frontend...","done":true}
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

#### Test Coverage

The test suite covers the following areas:

1. **Core Functionality Tests**:
   - Conversation upload and retrieval
   - User creation and management
   - Message embedding generation

2. **Retrieval Augmented Generation (RAG) Tests**:
   - Conversation RAG endpoint testing
   - Content retrieval based on semantic similarity
   - Edge cases and error handling

3. **Personality RAG Tests**:
   - Personality-enhanced RAG endpoint
   - Integration of user messages with personality profiles
   - Error handling and parameter customization

4. **Auto-Updating Embedding Tests**:
   - Embedding queue management
   - Batch processing of embedding updates
   - Database interaction and concurrency

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
- `OLLAMA_CHAT_MODEL`: Model name for chat functionality (default: llama3)
- `OLLAMA_EMBEDDING_MODEL`: Model name for embeddings (default: nomic-embed-text)
- `REDIS_URL`: Redis connection URL for caching (optional)
- `API_HOST`: Host to bind the API server
- `API_PORT`: Port to bind the API server
- `ENVIRONMENT`: Application environment (development, production)

## Personality Prompts

The system uses text prompt templates to generate personality profiles and simulate responses from users. These templates are stored in the `app/core/prompts/` directory:

- `personality_analysis.txt`: System prompt for analyzing user messages and generating a personality profile
- `personality_simulation.txt`: Template for simulating user responses based on their personality profile
- `description_template.txt`: Template for formatting the human-readable personality description

To modify how the AI interprets personalities or how it simulates user responses, you can edit these text files without changing any code.

## API Improvements

### Background Tasks

The API now processes intensive operations like personality profile generation in the background to prevent blocking concurrent requests. When you request profile generation, the API:

1. Immediately returns a response indicating the task has started
2. Continues processing in the background
3. Allows you to check the status through the `/personalities/users/{username}/profile-status` endpoint

Example workflow:

```
# Request profile generation
POST /personalities/users/john_doe/generate

# Check status of generation
GET /personalities/users/john_doe/profile-status

# Once complete, retrieve the full profile
GET /personalities/users/john_doe?active_only=true
```

This allows the API to remain responsive even when handling computationally intensive tasks. 