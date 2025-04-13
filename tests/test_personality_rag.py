import pytest
import json
from fastapi.testclient import TestClient
from unittest import mock
import math
import os
import httpx
import numpy as np

from app.services.embedding_service import EmbeddingService, EmbeddingAutoUpdater
from app.services.personality_service import PersonalityService

# Sample test data
TEST_CONVERSATION = {
    "source": "slack",
    "messages": [
        {
            "user": "Hardik",
            "timestamp": "1716292517.637399",
            "message": "I prefer React over Angular for frontend development."
        },
        {
            "user": "Hardik",
            "timestamp": "1716292510.807159",
            "message": "TypeScript gives better type safety than plain JavaScript."
        },
        {
            "user": "Murali",
            "timestamp": "1716292490.830249",
            "message": "What's your opinion on using React for this project?"
        },
        {
            "user": "Hardik",
            "timestamp": "1716292480.830249",
            "message": "I think we should use Next.js for our new website."
        },
        {
            "user": "Hardik",
            "timestamp": "1716292470.830249",
            "message": "I've been using React for several projects and it's working great for us."
        },
        {
            "user": "Murali",
            "timestamp": "1716292460.830249",
            "message": "What tech stack do you recommend?"
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

# Sample personality profile traits
SAMPLE_PERSONALITY_TRAITS = {
    "traits": {
        "openness": 8,
        "conscientiousness": 7,
        "extraversion": 6,
        "agreeableness": 7,
        "neuroticism": 4
    },
    "communication_style": {
        "clarity": "high",
        "directness": "medium",
        "technical_level": "expert",
        "formality": "casual"
    },
    "interests": ["programming", "web development", "technology", "React"],
    "values": ["efficiency", "innovation", "pragmatism"],
    "summary": "Technically skilled developer who values practical solutions."
}

# Mock embedding function to avoid calling Ollama
async def mock_generate_embedding(text):
    """Mock function to generate a fake embedding vector based on text content"""
    # Return a vector of 768 dimensions with values derived from text hash
    import hashlib
    hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
    # Use the hash to create a somewhat unique embedding vector
    vector = [(((hash_val + i) % 1000) / 1000) for i in range(768)]
    return vector

# Mock for generate_profile
async def mock_generate_profile(user_id, db):
    """Mock for personality profile generation"""
    from app.models.personality import PersonalityProfile
    
    # Create a mock personality profile
    profile = PersonalityProfile(
        id=1,
        user_id=user_id,
        traits=SAMPLE_PERSONALITY_TRAITS,
        description="Hardik is a technically skilled developer who values practical solutions.",
        embedding=[0.1] * 768,
        message_count=5,
        system_prompt="You are simulating Hardik, a developer who prefers React and likes TypeScript.",
        is_active=True
    )
    
    return profile

# Mock Ollama API response for chat
async def mock_ollama_chat_response(*args, **kwargs):
    """Mock function for Ollama chat API responses"""
    request_json = kwargs.get('json', {})
    messages = request_json.get('messages', [])
    user_message = next((m['content'] for m in messages if m['role'] == 'user'), "")
    
    # Default answer if nothing specific matches
    answer = "I don't have a strong opinion on that."
    
    # Check for specific queries in the user message
    if "react" in user_message.lower():
        answer = "I think React would be a solid choice for the frontend. I have previously worked with React and found it quite efficient for component-based UIs. As I mentioned before, its ecosystem is mature and there's plenty of support available."
    elif "typescript" in user_message.lower():
        answer = "I'm a big fan of TypeScript for its type safety features. It helps catch errors during development rather than at runtime."
    
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "message": {
            "content": answer
        }
    }
    
    return mock_response

# Mock for process queue
async def mock_process_queue(*args, **kwargs):
    """Mock for the embedding auto-updater's _process_queue method"""
    return None

@pytest.fixture
def mock_embedding_service():
    """Patch the embedding service for testing"""
    with mock.patch.object(EmbeddingService, 'generate_embedding', side_effect=mock_generate_embedding):
        yield

@pytest.fixture
def mock_personality_service():
    """Patch the personality service for testing"""
    with mock.patch.object(PersonalityService, 'generate_profile', side_effect=mock_generate_profile):
        yield

@pytest.fixture
def mock_ollama_api():
    """Patch the httpx.AsyncClient.post method to mock Ollama API calls"""
    with mock.patch.object(httpx.AsyncClient, 'post', side_effect=mock_ollama_chat_response):
        yield

@pytest.fixture
def mock_auto_updater():
    """Patch the embedding auto-updater for testing"""
    with mock.patch.object(EmbeddingAutoUpdater, '_process_queue', side_effect=mock_process_queue):
        yield

def test_setup_conversation_and_profile(client, mock_embedding_service, mock_personality_service, mock_auto_updater):
    """Test creating a conversation and generating a personality profile"""
    # First, create a conversation with test data
    response = client.post("/conversations/", json=TEST_CONVERSATION)
    assert response.status_code == 201
    
    # Generate a personality profile for the user
    profile_response = client.post("/personalities/users/Hardik/generate")
    assert profile_response.status_code == 200
    
    # Verify the profile was created
    profile_data = profile_response.json()
    assert profile_data["user_id"] is not None
    assert "traits" in profile_data
    assert profile_data["is_active"] is True

def test_personality_rag_endpoint(client, mock_embedding_service, mock_personality_service, mock_ollama_api, mock_auto_updater):
    """Test the personality RAG endpoint"""
    # First, set up conversation and profile
    client.post("/conversations/", json=TEST_CONVERSATION)
    client.post("/personalities/users/Hardik/generate")
    
    # Test the personality RAG endpoint
    question = {"question": "What do you think about using React for this project?"}
    rag_response = client.post("/personalities/users/Hardik/ask/rag", json=question)
    assert rag_response.status_code == 200
    
    data = rag_response.json()
    assert data["question"] == question["question"]
    assert "answer" in data
    assert "React" in data["answer"]
    assert "username" in data
    assert data["username"] == "Hardik"
    assert "used_messages" in data
    assert data["used_messages"] > 0
    assert "relevant_context" in data
    assert len(data["relevant_context"]) > 0

def test_personality_rag_with_custom_params(client, mock_embedding_service, mock_personality_service, mock_ollama_api, mock_auto_updater):
    """Test the personality RAG endpoint with custom parameters"""
    # First, set up conversation and profile
    client.post("/conversations/", json=TEST_CONVERSATION)
    client.post("/personalities/users/Hardik/generate")
    
    # Test with custom parameters
    question = {"question": "What do you think about TypeScript?"}
    rag_response = client.post("/personalities/users/Hardik/ask/rag?max_context_messages=2&model=llama3", json=question)
    assert rag_response.status_code == 200
    
    data = rag_response.json()
    assert data["question"] == question["question"]
    assert "answer" in data
    assert "TypeScript" in data["answer"]
    assert "relevant_context" in data
    assert len(data["relevant_context"]) <= 2  # Should respect max_context_messages

def test_personality_rag_no_relevant_messages(client, mock_embedding_service, mock_personality_service, mock_ollama_api, mock_auto_updater):
    """Test the personality RAG endpoint when no relevant messages are found"""
    # First, set up conversation and profile
    client.post("/conversations/", json=TEST_CONVERSATION)
    client.post("/personalities/users/Hardik/generate")
    
    # Ask about something not in the messages
    question = {"question": "What's your favorite color?"}
    rag_response = client.post("/personalities/users/Hardik/ask/rag", json=question)
    assert rag_response.status_code == 200
    
    data = rag_response.json()
    assert data["question"] == question["question"]
    assert "answer" in data
    assert "used_messages" in data
    # The personality model should still generate a response even without relevant context
    assert data["answer"] is not None

def test_personality_rag_user_not_found(client, mock_embedding_service, mock_personality_service, mock_ollama_api, mock_auto_updater):
    """Test the personality RAG endpoint with a non-existent user"""
    # Try to use the endpoint with a user that doesn't exist
    question = {"question": "What do you think about React?"}
    rag_response = client.post("/personalities/users/NonExistentUser/ask/rag", json=question)
    assert rag_response.status_code == 404
    
    data = rag_response.json()
    assert "detail" in data
    assert "not found" in data["detail"]

def test_personality_rag_no_profile(client, mock_embedding_service, mock_ollama_api, mock_auto_updater):
    """Test the personality RAG endpoint when no personality profile exists"""
    # Create a conversation but don't generate a profile
    client.post("/conversations/", json=TEST_CONVERSATION)
    
    # Try to use the endpoint
    question = {"question": "What do you think about React?"}
    rag_response = client.post("/personalities/users/Hardik/ask/rag", json=question)
    assert rag_response.status_code == 404
    
    data = rag_response.json()
    assert "detail" in data
    assert "No active personality profile found" in data["detail"]

def test_personality_rag_api_error(client, mock_embedding_service, mock_personality_service, mock_auto_updater):
    """Test error handling in the personality RAG endpoint"""
    # First, set up conversation and profile
    client.post("/conversations/", json=TEST_CONVERSATION)
    client.post("/personalities/users/Hardik/generate")
    
    # Mock an error in the Ollama API
    with mock.patch.object(httpx.AsyncClient, 'post', side_effect=Exception("API Error")):
        question = {"question": "What do you think about React?"}
        rag_response = client.post("/personalities/users/Hardik/ask/rag", json=question)
        assert rag_response.status_code == 500
        
        data = rag_response.json()
        assert "detail" in data
        assert "Error generating response" in data["detail"] 