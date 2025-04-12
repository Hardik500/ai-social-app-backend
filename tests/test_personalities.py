import pytest
import json
from fastapi.testclient import TestClient
from unittest import mock
import math
import copy

from app.services.embedding_service import EmbeddingService
from app.services.personality_service import PersonalityService
from app.main import app

# Import the mock_embedding_service fixture from test_conversations
from tests.test_conversations import mock_embedding_service

# Sample response for personality analysis
MOCK_PERSONALITY_ANALYSIS = {
    "traits": {
        "openness": 8,
        "conscientiousness": 7,
        "extraversion": 6,
        "agreeableness": 8,
        "neuroticism": 4
    },
    "communication_style": {
        "formality": "moderate",
        "directness": "high",
        "verbosity": "medium"
    },
    "interests": ["technology", "programming", "collaboration"],
    "values": ["efficiency", "clarity", "teamwork"],
    "summary": "A collaborative, tech-focused individual who values clear and direct communication. Shows high openness to new ideas and strong agreeableness in team settings."
}

# Sample response for personality-based answers
MOCK_RESPONSE = "I would definitely suggest using FastAPI for this project. It's much more efficient and has better type-checking capabilities."

# Create a mock for the personality service
@pytest.fixture
def mock_personality_service(monkeypatch):
    async def mock_generate_profile(user_id, db):
        from app.models.personality import PersonalityProfile
        
        profile = PersonalityProfile(
            id=1,
            user_id=user_id,
            traits=MOCK_PERSONALITY_ANALYSIS,
            description="Mock personality profile",
            embedding=[0.1] * 768,
            message_count=10,
            system_prompt="You are roleplaying as a user",
            is_active=True
        )
        
        # Set this in the database mock
        profile.id = 1
        profile.generated_at = "2023-01-01T00:00:00"
        profile.updated_at = None
        
        return profile
    
    async def mock_generate_response(user_id, question, db):
        return MOCK_RESPONSE
        
    monkeypatch.setattr(PersonalityService, "generate_profile", mock_generate_profile)
    monkeypatch.setattr(PersonalityService, "generate_response", mock_generate_response)

# Sample test data from the conversations test
from tests.test_conversations import TEST_CONVERSATION

# Create test client
client = TestClient(app)

def test_generate_personality_profile(mock_embedding_service, mock_personality_service):
    """Test generating a personality profile for a user"""
    # First, create a conversation to have some user data
    client.post("/conversations/", json=TEST_CONVERSATION)
    
    # Now generate a personality profile for the user
    username = TEST_CONVERSATION["primary_user_info"]["username"]
    response = client.post(f"/personalities/users/{username}/generate")
    
    assert response.status_code == 200
    data = response.json()
    
    # Check the profile data
    assert data["user_id"] > 0
    assert data["traits"] == MOCK_PERSONALITY_ANALYSIS
    assert data["is_active"] is True
    assert data["message_count"] == 10
    assert "system_prompt" in data

def test_get_user_profiles(mock_embedding_service, mock_personality_service):
    """Test retrieving personality profiles for a user"""
    # First, create a conversation and profile
    client.post("/conversations/", json=TEST_CONVERSATION)
    username = TEST_CONVERSATION["primary_user_info"]["username"]
    client.post(f"/personalities/users/{username}/generate")
    
    # Get all profiles for the user
    response = client.get(f"/personalities/users/{username}")
    assert response.status_code == 200
    
    data = response.json()
    assert len(data) >= 1
    assert data[0]["user_id"] > 0
    
    # Test with active_only parameter
    response = client.get(f"/personalities/users/{username}?active_only=true")
    assert response.status_code == 200
    
    data = response.json()
    assert len(data) >= 1
    assert data[0]["is_active"] is True

def test_get_profile_by_id(mock_embedding_service, mock_personality_service):
    """Test retrieving a specific personality profile by ID"""
    # First, create a conversation and profile
    client.post("/conversations/", json=TEST_CONVERSATION)
    username = TEST_CONVERSATION["primary_user_info"]["username"]
    profile_response = client.post(f"/personalities/users/{username}/generate")
    profile_id = profile_response.json()["id"]
    
    # Get the profile by ID
    response = client.get(f"/personalities/profiles/{profile_id}")
    assert response.status_code == 200
    
    data = response.json()
    assert data["id"] == profile_id
    assert data["user_id"] > 0

def test_ask_question(mock_embedding_service, mock_personality_service):
    """Test asking a question to a user based on their personality profile"""
    # First, create a conversation and profile
    client.post("/conversations/", json=TEST_CONVERSATION)
    username = TEST_CONVERSATION["primary_user_info"]["username"]
    client.post(f"/personalities/users/{username}/generate")
    
    # Ask a question
    question = "What framework would you recommend for a new API project?"
    response = client.post(
        f"/personalities/users/{username}/ask", 
        json={"question": question}
    )
    
    assert response.status_code == 200
    
    data = response.json()
    assert data["question"] == question
    assert data["answer"] == MOCK_RESPONSE
    assert data["username"] == username

def test_user_not_found(mock_embedding_service, mock_personality_service):
    """Test error handling when user is not found"""
    response = client.post("/personalities/users/nonexistent/generate")
    assert response.status_code == 404
    
    response = client.get("/personalities/users/nonexistent")
    assert response.status_code == 404
    
    response = client.post(
        "/personalities/users/nonexistent/ask", 
        json={"question": "Hello?"}
    )
    assert response.status_code == 404 