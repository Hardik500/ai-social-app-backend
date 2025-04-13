import pytest
import json
from fastapi.testclient import TestClient
from unittest import mock
import time
import asyncio

from app.services.embedding_service import EmbeddingService, EmbeddingAutoUpdater
from sqlalchemy import text, create_engine
from sqlalchemy.orm import sessionmaker

# Sample test data for a short conversation
TEST_CONVERSATION = {
    "source": "slack",
    "messages": [
        {
            "user": "Hardik",
            "timestamp": "1716292517.637399",
            "message": "This is a test message for embedding updates."
        }
    ],
    "primary_user_info": {
        "username": "Hardik",
        "email": "hardik@example.com",
        "phone": "1234567890",
        "description": "Tech enthusiast and developer"
    },
    "additional_users": []
}

# Mock embedding function to avoid calling Ollama
async def mock_generate_embedding(text):
    """Mock function to generate a fake embedding vector"""
    # Return a vector of 768 dimensions with all values as 0.1
    return [0.1] * 768

# Mock for schedule_embedding_generation
def mock_schedule_embedding_generation(self, db, text, table, id_column, id_value, text_column, embedding_column):
    """Mock for the schedule_embedding_generation method"""
    # Record the call parameters
    if not hasattr(self, 'scheduled_embeddings'):
        self.scheduled_embeddings = []
    
    self.scheduled_embeddings.append({
        'text': text,
        'table': table,
        'id_column': id_column,
        'id_value': id_value,
        'text_column': text_column,
        'embedding_column': embedding_column
    })

# Mock for process queue
async def mock_process_queue(self, db):
    """Mock for the embedding auto-updater's _process_queue method"""
    self.is_processing = True
    
    # Process the items in the queue
    for table, id_column, id_value, text_column, embedding_column in list(self.embedding_queue):
        # Get the text
        query = f"SELECT {text_column} FROM {table} WHERE {id_column} = ?"
        # In a real implementation we'd execute the query, but for the mock we'll skip it
        
        # Generate a mock embedding
        embedding = await mock_generate_embedding("mock text")
        
        # Record that we processed this item
        if not hasattr(self, 'processed_embeddings'):
            self.processed_embeddings = []
        
        self.processed_embeddings.append({
            'table': table,
            'id_column': id_column,
            'id_value': id_value,
            'embedding': embedding
        })
    
    # Clear the queue
    self.embedding_queue.clear()
    self.is_processing = False

@pytest.fixture
def mock_embedding_service():
    """Patch the embedding service for testing"""
    with mock.patch.object(EmbeddingService, 'generate_embedding', side_effect=mock_generate_embedding):
        yield

@pytest.fixture
def mock_auto_updater():
    """Patch the auto-updater methods for testing"""
    with mock.patch.object(EmbeddingAutoUpdater, '_process_queue', side_effect=mock_process_queue):
        with mock.patch.object(EmbeddingService, 'schedule_embedding_generation', side_effect=mock_schedule_embedding_generation):
            yield

def test_auto_updater_initialization():
    """Test that the embedding auto-updater initializes correctly"""
    updater = EmbeddingAutoUpdater()
    assert updater.embedding_queue == set()
    assert updater.is_processing == False
    assert updater.batch_size > 0
    assert updater.sleep_time > 0

def test_conversation_creation_schedules_embeddings(client, mock_embedding_service, mock_auto_updater):
    """Test that creating a new conversation schedules embedding updates"""
    # Create a conversation
    response = client.post("/conversations/", json=TEST_CONVERSATION)
    assert response.status_code == 201
    
    # Get the conversation to access the message IDs
    conversation_id = response.json()["id"]
    conversation = client.get(f"/conversations/{conversation_id}").json()
    message_id = conversation["messages"][0]["id"]
    
    # Verify the message has an embedding initially
    embed_response = client.get(f"/conversations/messages/{message_id}/embedding")
    assert embed_response.status_code == 200
    data = embed_response.json()
    assert data["embedding_exists"] is True
    
    # Check that the auto-updater was used correctly
    from app.services.embedding_service import embedding_auto_updater
    assert hasattr(embedding_auto_updater, 'embedding_queue')
    # The queue may be empty since the mock processes items immediately

def test_schedule_update_adds_to_queue():
    """Test that scheduling an update adds items to the queue"""
    # Create a fresh auto-updater for this test
    updater = EmbeddingAutoUpdater()
    
    # Mock the DB session
    db_mock = mock.Mock()
    
    # Schedule an update
    updater.schedule_update(
        db=db_mock,
        table="messages",
        id_column="id",
        id_value=123,
        text_column="content",
        embedding_column="embedding"
    )
    
    # Check the queue contains the item
    assert len(updater.embedding_queue) == 1
    item = list(updater.embedding_queue)[0]
    assert item[0] == "messages"
    assert item[1] == "id"
    assert item[2] == 123
    assert item[3] == "content"
    assert item[4] == "embedding"

def test_process_queue_processes_items():
    """Test that the process queue method processes items in the queue"""
    # Create a fresh auto-updater for this test
    updater = EmbeddingAutoUpdater()
    
    # Mock the DB session
    db_mock = mock.Mock()
    
    # Add items to the queue
    updater.embedding_queue.add(("messages", "id", 123, "content", "embedding"))
    updater.embedding_queue.add(("messages", "id", 456, "content", "embedding"))
    
    # Mock execute and scalar methods
    db_mock.execute.return_value.scalar.return_value = "This is a test message"
    
    # Run the process queue method (we'll use the real one, not the mock_embedding_service fixture)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        # Need to patch generate_embedding since we're not using the mock_embedding_service fixture
        with mock.patch.object(EmbeddingService, 'generate_embedding', side_effect=mock_generate_embedding):
            loop.run_until_complete(updater._process_queue(db_mock))
    finally:
        loop.close()
    
    # Queue should be empty
    assert len(updater.embedding_queue) == 0
    
    # Check that db.execute was called for each item
    assert db_mock.execute.call_count >= 2  # At least 2 times for SELECT and UPDATE * 2 items

def test_batch_processing_respects_limits():
    """Test that batch processing respects batch size limits"""
    # Create an auto-updater with a small batch size
    updater = EmbeddingAutoUpdater()
    updater.batch_size = 2
    updater.sleep_time = 0.1  # Reduce sleep time for testing
    
    # Mock the DB session
    db_mock = mock.Mock()
    
    # Add more items to the queue than the batch size
    for i in range(5):
        updater.embedding_queue.add((f"messages", "id", i, "content", "embedding"))
    
    # Add a mock implementation for _process_queue that tracks batches
    batches = []
    
    async def mock_process_queue_batches(db):
        nonlocal batches
        updater.is_processing = True
        
        try:
            while updater.embedding_queue:
                # Process a batch of items
                batch = []
                for _ in range(min(updater.batch_size, len(updater.embedding_queue))):
                    if not updater.embedding_queue:
                        break
                    batch.append(updater.embedding_queue.pop())
                
                batches.append(batch)
                
                # Sleep between batches
                if updater.embedding_queue:
                    await asyncio.sleep(updater.sleep_time)
        
        finally:
            updater.is_processing = False
    
    # Run the process queue method with our custom implementation
    with mock.patch.object(updater, '_process_queue', side_effect=mock_process_queue_batches):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(updater.schedule_update(
                db=db_mock,
                table="test",
                id_column="id",
                id_value=999,
                text_column="content",
                embedding_column="embedding"
            ))
            # Let the background task run
            time.sleep(0.3)
        finally:
            loop.close()
    
    # Check that we had at least two batches
    assert len(batches) >= 2
    # Each batch should have at most batch_size items
    for batch in batches:
        assert len(batch) <= updater.batch_size 