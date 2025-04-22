import json
import os
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.main import app
from app.services.ingestion_service import IngestionService
from app.db.database import Base, engine, get_db
from app.models.user import User
from app.models.conversation import Conversation, Message

# Test client
client = TestClient(app)

# Sample data
SAMPLE_HAR_DATA = {
    "log": {
        "entries": [
            {
                "response": {
                    "content": {
                        "text": json.dumps({
                            "messages": [
                                {
                                    "user": "U055WM6DTJL",
                                    "type": "message",
                                    "ts": "1744947001.192049",
                                    "text": "Sample message from Slack"
                                },
                                {
                                    "user": "U03HPSXQXHC",
                                    "type": "message",
                                    "ts": "1744865828.558479",
                                    "text": "Reply to the sample message"
                                }
                            ]
                        })
                    }
                }
            }
        ]
    }
}

SAMPLE_PRIMARY_USER = {
    "username": "Hardik",
    "email": "hkhandelwal@example.com",
    "phone": "+9191234567890",
    "description": "Test user"
}

SAMPLE_ADDITIONAL_USERS = [
    {
        "username": "Murali",
        "email": "murali.v@example.com",
        "phone": "+9198765432109",
        "description": "Additional test user"
    }
]

SAMPLE_USER_MAPPING = {
    "U055WM6DTJL": "Hardik",
    "U03HPSXQXHC": "Murali"
}

@pytest.fixture
def db():
    # Drop and recreate the tables for testing
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    
    # Create a new session for tests
    db = next(get_db())
    
    try:
        yield db
    finally:
        db.close()

def test_ingestion_service(db: Session):
    """Test the IngestionService with sample data."""
    # Create service
    service = IngestionService(db)
    
    # Test ingestion
    result = service.ingest_data(
        source_type="slack_har",
        source_data=SAMPLE_HAR_DATA,
        primary_user_info=SAMPLE_PRIMARY_USER,
        additional_users=SAMPLE_ADDITIONAL_USERS,
        user_mapping=SAMPLE_USER_MAPPING
    )
    
    # Check result
    assert result["status"] == "success"
    assert result["messages_imported"] > 0
    assert "warnings" not in result  # No warnings with proper mapping
    
    # Check database
    conversation = db.query(Conversation).first()
    assert conversation is not None
    assert conversation.source == "slack"
    
    messages = db.query(Message).all()
    assert len(messages) == result["messages_imported"]
    
    users = db.query(User).all()
    assert len(users) >= 2  # Primary user + at least one message sender
    
    # Verify the users are mapped correctly
    hardik = db.query(User).filter(User.username == "Hardik").first()
    murali = db.query(User).filter(User.username == "Murali").first()
    assert hardik is not None
    assert murali is not None
    
    # Verify at least one message from each user exists
    hardik_messages = db.query(Message).filter(Message.user_id == hardik.id).count()
    murali_messages = db.query(Message).filter(Message.user_id == murali.id).count()
    assert hardik_messages > 0
    assert murali_messages > 0

def test_ingestion_service_without_mapping(db: Session):
    """Test the IngestionService without user mapping."""
    # Create service
    service = IngestionService(db)
    
    # Test ingestion without mapping
    result = service.ingest_data(
        source_type="slack_har",
        source_data=SAMPLE_HAR_DATA,
        primary_user_info=SAMPLE_PRIMARY_USER,
        additional_users=SAMPLE_ADDITIONAL_USERS
    )
    
    # Check result
    assert result["status"] == "success"
    assert result["messages_imported"] > 0
    assert "warnings" in result  # Should have warnings about unknown users
    assert "unknown_users" in result["warnings"]
    assert len(result["warnings"]["unknown_users"]) > 0

def test_api_endpoint():
    """Test the API endpoint for ingestion."""
    # Convert data to proper format for API
    response = client.post(
        "/ingestion/test",
        json={
            "source_type": "slack_har",
            "source_data": SAMPLE_HAR_DATA,
            "primary_user_info": SAMPLE_PRIMARY_USER,
            "additional_users": SAMPLE_ADDITIONAL_USERS,
            "user_mapping": SAMPLE_USER_MAPPING
        }
    )
    
    # Check response
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "success"
    assert result["messages_imported"] > 0
    assert "warnings" not in result  # No warnings with proper mapping

if __name__ == "__main__":
    # Run test directly
    pytest.main(["-xvs", __file__]) 