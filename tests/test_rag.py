import pytest
import json
from fastapi.testclient import TestClient
from unittest import mock
import math
import os
import httpx

from app.services.embedding_service import EmbeddingService, EmbeddingAutoUpdater

# Sample test data
TEST_CONVERSATION = {
    "source": "slack",
    "messages": [
        {
            "user": "Hardik",
            "timestamp": "1716292517.637399",
            "message": "We need to finish the project by March 15th, 2023."
        },
        {
            "user": "Murali",
            "timestamp": "1716292510.807159",
            "message": "Don't forget about the deadline next month!"
        },
        {
            "user": "Sarah",
            "timestamp": "1716292490.830249",
            "message": "The client expects all deliverables by mid-March."
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
        },
        {
            "username": "Sarah",
            "email": "sarah@example.com",
            "phone": "5555555555",
            "description": "Project manager"
        }
    ]
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

# Mock Ollama API response for chat
async def mock_ollama_chat_response(*args, **kwargs):
    """Mock function for Ollama chat API responses"""
    request_json = kwargs.get('json', {})
    messages = request_json.get('messages', [])
    user_message = next((m['content'] for m in messages if m['role'] == 'user'), "")
    
    # Default answer if nothing specific matches
    answer = "Based on the provided context, I don't have enough information to answer that question."
    
    # Check for specific queries in the user message
    if "deadline" in user_message.lower():
        answer = "Based on the provided context, the project deadline is March 15th, 2023."
    elif "deliverable" in user_message.lower():
        answer = "All deliverables are expected by mid-March according to the context."
    
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
def mock_ollama_api():
    """Patch the httpx.AsyncClient.post method to mock Ollama API calls"""
    with mock.patch.object(httpx.AsyncClient, 'post', side_effect=mock_ollama_chat_response):
        yield

@pytest.fixture
def mock_auto_updater():
    """Patch the embedding auto-updater for testing"""
    with mock.patch.object(EmbeddingAutoUpdater, '_process_queue', side_effect=mock_process_queue):
        yield

def test_conversation_rag_endpoint(client, mock_embedding_service, mock_ollama_api, mock_auto_updater):
    """Test the conversation RAG endpoint with similar messages retrieval"""
    # First, create a conversation with test data
    response = client.post("/conversations/", json=TEST_CONVERSATION)
    assert response.status_code == 201
    
    # Test the RAG endpoint with a deadline question
    query = "What is the project deadline?"
    rag_response = client.post(f"/conversations/rag?query={query}")
    assert rag_response.status_code == 200
    
    data = rag_response.json()
    assert data["query"] == query
    assert "answer" in data
    assert "Based on the provided context, the project deadline is March 15th, 2023." in data["answer"]
    assert "context_used" in data
    assert len(data["context_used"]) > 0
    assert "model_used" in data

def test_conversation_rag_no_similar_messages(client, mock_embedding_service, mock_ollama_api, mock_auto_updater):
    """Test the RAG endpoint when no similar messages are found"""
    # Test with an empty database
    query = "What is the project budget?"
    rag_response = client.post(f"/conversations/rag?query={query}")
    assert rag_response.status_code == 200
    
    data = rag_response.json()
    assert data["query"] == query
    assert "answer" in data
    assert "couldn't find any relevant information" in data["answer"].lower()
    assert "context_used" in data
    assert len(data["context_used"]) == 0

def test_conversation_rag_custom_params(client, mock_embedding_service, mock_ollama_api, mock_auto_updater):
    """Test the RAG endpoint with custom parameters"""
    # First, create a conversation with test data
    response = client.post("/conversations/", json=TEST_CONVERSATION)
    assert response.status_code == 201
    
    # Test with custom parameters
    query = "What are the deliverables?"
    rag_response = client.post(f"/conversations/rag?query={query}&max_context_messages=2&model=llama3")
    assert rag_response.status_code == 200
    
    data = rag_response.json()
    assert data["query"] == query
    assert "model_used" in data
    assert data["model_used"] == "llama3"  # Should use the specified model
    assert len(data["context_used"]) <= 2  # Should respect max_context_messages

def test_conversation_rag_error_handling(client, mock_embedding_service, mock_auto_updater):
    """Test error handling in the RAG endpoint"""
    # Mock an error in the Ollama API
    with mock.patch.object(httpx.AsyncClient, 'post', side_effect=Exception("API Error")):
        query = "What is the project deadline?"
        rag_response = client.post(f"/conversations/rag?query={query}")
        assert rag_response.status_code == 200  # Should still return 200 with error info
        
        data = rag_response.json()
        assert data["query"] == query
        assert "error" in data
        assert "API Error" in data["error"]
        assert data["answer"] is None

def test_auto_updating_embeddings(client, mock_embedding_service, mock_auto_updater):
    """Test that embeddings are auto-updated when new messages are added"""
    # Create a conversation
    response = client.post("/conversations/", json=TEST_CONVERSATION)
    assert response.status_code == 201
    
    # Check message embeddings
    conversation = response.json()
    for message in conversation["messages"]:
        message_id = message["id"]
        embed_response = client.get(f"/conversations/messages/{message_id}/embedding")
        assert embed_response.status_code == 200
        data = embed_response.json()
        assert data["embedding_exists"] is True 