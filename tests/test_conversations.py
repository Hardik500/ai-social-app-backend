import pytest
import json
from fastapi.testclient import TestClient
from unittest import mock
import math
import copy

from app.services.embedding_service import EmbeddingService

# Sample test data
TEST_CONVERSATION = {
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
        }
    ],
    "user_info": {
        "username": "Hardik",
        "email": "hardik@example.com",
        "phone": "1234567890",
        "description": "Tech enthusiast and developer"
    }
}

# Mock embedding function to avoid calling Ollama
async def mock_generate_embedding(text):
    """Mock function to generate a fake embedding vector"""
    # Return a vector of 768 dimensions with all values as 0.1
    return [0.1] * 768

@pytest.fixture
def mock_embedding_service():
    """Patch the embedding service for testing"""
    with mock.patch.object(EmbeddingService, 'generate_embedding', side_effect=mock_generate_embedding):
        yield

def test_create_conversation(client, mock_embedding_service):
    """Test creating a new conversation"""
    response = client.post("/conversations/", json=TEST_CONVERSATION)
    assert response.status_code == 201
    
    data = response.json()
    assert data["source"] == "slack"
    assert len(data["messages"]) == 2
    assert data["messages"][0]["content"] == "Got it, then lets have it offline next week only"

def test_get_all_conversations(client, mock_embedding_service):
    """Test retrieving all conversations"""
    # First, create a conversation
    client.post("/conversations/", json=TEST_CONVERSATION)
    
    # Then retrieve all conversations
    response = client.get("/conversations/")
    assert response.status_code == 200
    
    data = response.json()
    assert len(data) >= 1
    assert data[0]["source"] == "slack"

def test_get_conversation_by_id(client, mock_embedding_service):
    """Test retrieving a specific conversation by ID"""
    # First, create a conversation
    create_response = client.post("/conversations/", json=TEST_CONVERSATION)
    conversation_id = create_response.json()["id"]
    
    # Then retrieve the specific conversation
    response = client.get(f"/conversations/{conversation_id}")
    assert response.status_code == 200
    
    data = response.json()
    assert data["id"] == conversation_id
    assert data["source"] == "slack"
    assert len(data["messages"]) == 2

def test_get_message_embedding(client, mock_embedding_service):
    """Test retrieving a message's embedding"""
    # First, create a conversation
    create_response = client.post("/conversations/", json=TEST_CONVERSATION)
    conversation_id = create_response.json()["id"]
    
    # Get the conversation to access the message IDs
    conversation = client.get(f"/conversations/{conversation_id}").json()
    message_id = conversation["messages"][0]["id"]
    
    # Then retrieve the embedding
    response = client.get(f"/conversations/messages/{message_id}/embedding")
    assert response.status_code == 200
    
    data = response.json()
    assert data["id"] == message_id
    assert data["embedding_exists"] == True
    assert data["embedding_dimensions"] == 768
    assert len(data["embedding_sample"]) == 5
    # Use approximate comparison for floating point values
    assert math.isclose(data["embedding_sample"][0], 0.1, rel_tol=1e-5)

def test_search_similar_messages(client, mock_embedding_service):
    """Test searching for similar messages"""
    # First, create a conversation
    client.post("/conversations/", json=TEST_CONVERSATION)
    
    # Then search for similar messages
    query = "meet next week"
    response = client.get(f"/conversations/search/similar?query={query}")
    assert response.status_code == 200
    
    data = response.json()
    assert data["query"] == query
    assert len(data["similar_messages"]) > 0
    
    # Since our mock returns the same embedding for all texts,
    # similarity scores should be close to 1.0 (perfect match)
    for message in data["similar_messages"]:
        assert math.isclose(message["similarity_score"], 1.0, rel_tol=1e-5)

def test_health_check(client):
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_duplicate_conversation_detection(client, mock_embedding_service):
    """Test that duplicate conversations are detected correctly"""
    # First, create a conversation
    response1 = client.post("/conversations/", json=TEST_CONVERSATION)
    assert response1.status_code == 201
    
    # Try to create the exact same conversation again
    response2 = client.post("/conversations/", json=TEST_CONVERSATION)
    assert response2.status_code == 200  # 200 instead of 201 for duplicates
    
    data = response2.json()
    assert "duplicate_detection" in data
    assert data["duplicate_detection"]["is_duplicate"] == True
    assert data["duplicate_detection"]["match_percentage"] >= 50  # Should have at least 50% match
    
    # Verify the returned conversation is the existing one
    assert data["id"] == response1.json()["id"]

def test_similar_but_different_conversations(client, mock_embedding_service):
    """Test that similar but sufficiently different conversations are not marked as duplicates"""
    # First, create a conversation
    response1 = client.post("/conversations/", json=TEST_CONVERSATION)
    assert response1.status_code == 201
    
    # Create a conversation with the same structure but different timestamps (shifted by 1 day)
    different_conversation = copy.deepcopy(TEST_CONVERSATION)
    
    # Shift timestamps by 1 day (86400 seconds)
    for msg in different_conversation["messages"]:
        timestamp_parts = msg["timestamp"].split(".")
        new_timestamp = str(float(timestamp_parts[0]) + 86400)
        if len(timestamp_parts) > 1:
            new_timestamp += "." + timestamp_parts[1]
        msg["timestamp"] = new_timestamp
    
    # Try to create the modified conversation
    response2 = client.post("/conversations/", json=different_conversation)
    assert response2.status_code == 201  # Should be created as new
    
    # The IDs should be different
    assert response1.json()["id"] != response2.json()["id"]

def test_skip_duplicate_detection(client, mock_embedding_service):
    """Test that duplicate detection can be bypassed with skip_duplicate_check parameter"""
    # First, create a conversation
    response1 = client.post("/conversations/", json=TEST_CONVERSATION)
    assert response1.status_code == 201
    
    # Try to create the exact same conversation again but skip duplicate check
    response2 = client.post("/conversations/?skip_duplicate_check=true", json=TEST_CONVERSATION)
    assert response2.status_code == 201  # Should create a new conversation
    
    # The IDs should be different
    assert response1.json()["id"] != response2.json()["id"] 