import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import sys
import numpy as np
from unittest import mock

from app.main import app
from app.db.database import Base, get_db
import pgvector.sqlalchemy  # Import to register Vector type

# Use in-memory SQLite for testing
TEST_DATABASE_URL = "sqlite:///:memory:"

# Create a mock Vector type for SQLite
class MockVector(list):
    def __init__(self, data):
        super().__init__(data)

# Add vector compatibility to SQLite for testing
@pytest.fixture
def mock_pgvector():
    # Mock Vector class to work with SQLite
    original_vector = pgvector.sqlalchemy.Vector
    
    # Create a patch for the Vector class
    vector_patch = mock.patch('pgvector.sqlalchemy.Vector', side_effect=lambda dim: original_vector(dim))
    
    # Start the patch
    vector_patch.start()
    
    # Add SQLite compatibility for Vector operations
    def sqlite_vector_adapter(vector):
        if vector is None:
            return None
        return np.array(vector).tobytes()
    
    def sqlite_vector_converter(raw):
        if raw is None:
            return None
        arr = np.frombuffer(raw, dtype=np.float32)
        return list(arr)
    
    # Hook for the patch teardown
    yield
    
    # Stop the patch
    vector_patch.stop()

@pytest.fixture
def test_db(mock_pgvector):
    # Create the SQLite engine with SQLAlchemy for testing
    engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Create the tables
    Base.metadata.create_all(bind=engine)
    
    # Set up SQLite to handle vector data
    @event.listens_for(engine, "connect")
    def do_connect(dbapi_connection, connection_record):
        # Enable math functions in SQLite for cosine similarity
        dbapi_connection.create_function("power", 2, lambda x, y: x ** y)
        dbapi_connection.create_function("sqrt", 1, lambda x: x ** 0.5)
        
        # Register custom functions for vector operations
        dbapi_connection.create_function("dot_product", 2, lambda a, b: np.dot(a, b))
        dbapi_connection.create_function("vector_norm", 1, lambda a: np.linalg.norm(a))
    
    # Dependency override
    def override_get_db():
        try:
            db = TestingSessionLocal()
            yield db
        finally:
            db.close()
    
    app.dependency_overrides[get_db] = override_get_db
    
    yield  # This is where the testing happens
    
    # Drop all tables after test
    Base.metadata.drop_all(bind=engine)
    
@pytest.fixture
def client(test_db):
    with TestClient(app) as c:
        yield c 